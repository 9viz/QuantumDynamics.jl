"PLDM implementation using the Bargmann coherent states for the system."
module PLDM

using HDF5
using ..Utilities
using ..TTM
using ..SpectralDensities
using ..Systems, ..SolventsX
using Distributions: Rayleigh as RayleighDist, Normal, UnivariateDistribution
import OrdinaryDiffEq as ODE

const references = ""

"Abstract type for all the sampling methods of the system DOF."
abstract type PLDMSystemSampler end

abstract type Gaussian <: PLDMSystemSampler end
sys_distribution(::Type{Gaussian}) = Normal(0.0, 1/√2)

abstract type Rayleigh <: PLDMSystemSampler end
sys_distribution(::Type{Rayleigh}) = RayleighDist(1/√2)

function sampleα(::Type{Gaussian}, dist::Normal, d::Integer)
    α⁺ = rand(dist, d) .+ im * rand(dist, d)
    α⁻ = rand(dist, d) .+ im * rand(dist, d)
    α⁺, α⁻
end

function sampleα(::Type{Rayleigh}, dist::RayleighDist, d::Integer)
    r⁺ = rand(dist, d)
    r⁻ = rand(dist, d)
    θ⁺ = 2π * rand(d)
    θ⁻ = 2π * rand(d)
    r⁺ .* exp.(im * θ⁺), r⁻ .* exp.(im * θ⁻)
end

function sample_weight(ρ₀::AbstractMatrix{<:Complex},
                       α⁺::AbstractVector{<:Complex},
                       α⁻::AbstractVector{<:Complex})
    α⁺' * ρ₀ * α⁻
end



struct PLDMSysPhaseSpace <: SolventsX.PhaseSpace
    uf::Vector{<:Complex}
    vf::Vector{<:Complex}
    ub::Vector{<:Complex}
    vb::Vector{<:Complex}
end

struct type PLDMSystem <: Systems.CompositeSystem
    h::AbstractMatrix
    ρ₀::Union{Nothing,AbstractMatrix{<:Complex}}
    sampler::Type{<:PLDMSystemSampler}
    distα::UnivariateDistribution
    d::Integer
    bath::SolventsX.Solvent
    nsamples::Integer
end
function PLDMSystem(; sampler::Type{<:PLDMSystemSampler}, hamiltonian::AbstractMatrix,
                    ρ₀::Union{Nothing,AbstractMatrix{<:Complex}}, bath::SolventsX.Solvent,
                    nsamples::Integer)
    @assert nsamples == bath.nsamples
    d = size(hamiltonian, 1)
    PLDMSystem(hamiltonian, ρ₀, sampler, sys_distribution(sampler), d, bath, nsamples)
end

function Base.iterate(sys::PLDMSystem, state=1)
    state > sys.nsamples && return nothing

    bathps, _ = iterate(sys.bath)

    α⁺, α⁻ = sampleα(sampler, dist, sys.d)
    (PLDMSysPhaseSpace(α⁺, conj(α⁺), α⁻, conj(α⁻)), bathps), state+1
end
Base.eltype(::PLDMSystem) = PLDMSysPhaseSpace
Base.length(sys::PLDMSystem) = sys.nsamples



𝓈(v::AbstractVector{<:Complex}, u::AbstractVector{<:Complex}, s::AbstractVector) = sum(v .* s .* u)
𝓈̄(sps::PLDMSysPhaseSpace, s::AbstractVector) = 0.5 * (𝓈(sps.uf, sps.vf, s) + 𝓈(sps.ub, sps.vb, s))

"Calculate the force on the `i`th bath."
Fbath(sys::PLDMSystem, sps::PLDMSysPhaseSpace, q::AbstractVector{<:Real}, i::Integer) =
    -sys.bath.ω[i].^2 .* q .+ 𝓈̄(sps, sys.bath.s[i]) .* sys.bath.c[i]

reconstruct_bare_ρ(uf::AbstractVector{<:Complex}, vb::AbstractVector{<:Complex}) =
    uf * transpose(vb)

function update_dynmap!(U0e::AbstractMatrix{<:Complex},
                        bareρ::AbstractMatrix{<:Complex},
                        bareρ′::AbstractMatrix{<:Complex},
                        sys::PLDMSystem,
                        sps0::PLDMSysPhaseSpace)
    d = size(bareρ, 1)
    d² = size(U0e, 1)

    for i in 1:d
        ρ = zeros(ComplexF64, d,d)
        ρ[i,i] = 1.0
        w = sample_weight(ρ, sps0.uf, sps0.ub)
        w′ = sample_weight(ρ, sps0.ub, sps0.uf)
        U0e[:,i+(i-1)*d] = Utilities.density_matrix_to_vector(
            0.5 * (w * bareρ + w′ * bareρ′))

        for j in i+1:d
            ρ .= 0.0
            ρ[i,j] = 1.0
            w = sample_weight(ρ, sps0.uf, sps0.ub)
            w′ = sample_weight(ρ, sps0.ub, sps0.uf)
            n = (i-1)*d + j
            U0e[:,n] = Utilities.density_matrix_to_vector(
                0.5 * (w * bareρ + w′ * bareρ′))

            m = n + (j-i) * (d-1)
            U0e[1:d+1:d²,m] = conj(U0e[1:d+1:d²,n])

            for a = 1:d, b = a+1:d
                l = (b-1)*d + a
                k = (a-1)*d + b
                Uoe[k,m] = conj(U0e[l,n])
                Uoe[l,m] = conj(U0e[k,n])
            end
        end
    end
end



abstract type PLDMSolver end
abstract type RK4 <: PLDMSolver end

function ℋu(sys::PLDMSystem, v::AbstractVector{<:Complex}, bps::SolventsX.PhaseSpace)
    bs = sys.bath

    transpose(v) * sys.h +
        -mapreduce((b, x) -> sum(bs.c[b] .* x) * v .* bs.s[b], +, 1:bs.nbaths, bps.q) +
        mapreduce(b -> 0.5 * sum(bs.c[b].^2 ./ bs.ω[b].^2) * v .* bs.s[b].^2, +, 1:bs.nbaths)
end

function ℋv(sys::PLDMSystem, u::AbstractVector{<:Complex}. bps::SolventsX.PhaseSpace)
    bs = sys.bath

    sys.h * u +
        -mapreduce((b, x) -> sum(bs.c[b] .* x) * bs.s[b] .* u, 1, 1:bs.nbaths, bps.q) +
        mapreduce(b -> 0.5 * sum(bs.c[b].^2 ./ bs.ω[b].^2) * bs.s[b].^2 .* u, +, 1:bs.nbaths)
end

function propagate_α_xp!(du, u, p, t)
    sys, xis, pis = p
    d = sys.d

    uf, vf, ub, vb = u[1:d+1], u[d+1:2d], u[2d+1:3d], u[3d+1:4d]
    sps = PLDMSysPhaseSpace(uf, vf, ub, vb)
    bps = SolventsX.HarmonicPhaseSpaceX([ u[i] for i in xis ], [ u[i] for i in pis ])

    du[1:d] = -im * ℋv(sys, uf, bps)    # u̇⁺
    du[d+1:2d] = im * ℋu(sys, vf, bps)  # v̇⁺
    du[2d+1:3d] = -im * ℋ(sys, ub, bps) # u̇⁻
    du[3d+1:4d] = im * ℋ(sys, vb, bps)  # v̇⁻

    for n in 1:sys.bath.nbaths
        du[xis[n]] = bps.p[n]                     # ẋ
        du[pis[n]] = Fbath(sys, sps, bps.q[n], n) # ṗ
    end
end

function bathinds(sys::PLDMSystem)
    m = length.(sys.bath.ω)
    total = sum(m)
    xis = Vector{UnitRange{Integer}}(undef, sys.bath.nbaths)
    pis = Vector{UnitRange{Integer}}(undef, sys.bath.nbaths)

    prevstart = 4sys.d
    for (n,len) in enumerate(m)
        xis[n] = prevstart+1:prevstart+len
        pis[n] = total+prevstart+1:total+prevstart+len
        prevstart = prevstart+len
    end

    xis, pis
end

function build_dynmap_ρ(sol::ODE.ODESolution)
    sys, _ = sol.prob.p
    d = sys.d
    Nₜ = length(sol.u)

    α₀⁺ = sol.u[1][1:d]
    α₀⁻ = sol.u[1][2d+1:3d]
    sps0 = PLDMSysPhaseSpace(α₀⁺, conj(α₀⁺), α₀⁻, conj(α₀⁻))

    U0e = zeros(ComplexF64, Nₜ-1,d^2,d^2)
    if !isnothing(sys.ρ₀)
        ρ = zeros(ComplexF64, Nₜ,d,d)
        ρ[1,:,:] = (sample_weight(sys.ρ₀, α₀⁺, α₀⁻) * reconstruct_bare_ρ(α₀⁺, conj(α₀⁻)) +
                    sample_weight(sys.ρ₀, α₀⁻, α₀⁺) * reconstruct_bare_ρ(α₀⁻, conj(α₀⁺))) / 2
        ρ₀ᵥ = Utilities.density_matrix_to_vector(sys.ρ₀)
    end

    for t in 2:Nₜ
        sps = PLDMSysPhaseSpace(
            sol.u[t][1:d],
            sol.u[t][d+1:2d],
            sol.u[t][2d+1:3d],
            sol.u[t][3d+1:4d])
        bareρ = reconstruct_bare_ρ(sps.uf, sps.vb)
        bareρ′ = reconstruct_bare_ρ(sps.uf, sps.vb)
        update_dynmap!(view(U0e, t-1,:,:), bareρ, bareρ′, sys, sps0)

        if !isnothing(sys.ρ₀)
            ρ[t,:,:] = Utilities.density_matrix_vector_to_matrix(
                U0e[t-1,:,:] * ρ₀ᵥ)
        end
    end

    (U0e, isnothing(sys.ρ₀) ? nothing : ρ)
end

function propagate_trajectories(::Type{RK4}, sys::PLDMSystem, dt::Real, ntimes::Integer;
                                output::Union{Nothing,HDF5.Group}=nothing, verbose::Bool=false,
                                kwargs...)
    xis, pis = bathinds(sys)

    outputρ = if !isnothing(output) && haskey(kwargs, :outgroup)
        Utilities.create_and_select_group(output, kwargs[:outgroup])
    else
        nothing
    end

    if !isnothing(output)
        d² = sys.d^2
        Utilities.check_or_insert_value(output, "U0e", zeros(ComplexF64, ntimes,d²,d²))
        Utilities.check_or_insert_value(output, "T0e", zeros(ComplexF64, ntimes,d²,d²))
        Utilities.check_or_insert_value(output, "samples_done", 0)
        !isnothing(sys.ρ₀) && !isnothing(outputρ) &&
            Utilities.check_or_insert_value(outputρ, "rho", zeros(ComplexF64, ntimes+1,d,d))
    end

    probfn(p, i, _) = begin
        ps, _ = iterate(sys, i)
        sps, bps = ps
        ODE.remake(ρ, u0=vcat(sps.uf, sps.vf, sps.ub, sps.vb, bps.q..., bps.p...))
    end
    outputfn(sol, _) = (build_dynmap_ρ(sol), false)

    done = 0
    reducefn(data, us, I) = begin
        done += length(I)
        verbose && @info "Trajectories completed: $(done * 100 / length(sys))%"

        U0e = data[1] + sum(getindex.(us, 1))
        ρ = isnothing(sys.ρ₀) ? nothing : data[2] + sum(getindex.(us, 2))

        if !isnothing(output)
            output["U0e"][:,:,:] = U0e / done
            output["T0e"][:,:,:] = TTM.get_Ts(U0e / done)
            delete_object(output, "samples_done")
            output["samples_done"] = done
            flush(output)
        end

        if !isnothing(ρ) && !isnothing(outputρ)
            outputρ["rho"][:,:,:] = ρ / done
            flush(outputρ)
        end

        ((U0e, ρ), false)
    end

    ensemble = ODE.EnsembleProblem(
    ODE.ODEProblem(propagate_α_xp!,
                   zeros(4d+2sum(length.(sys.bath.ω))),
                   (0.0, ntimes*dt),
                   (sys, xis, pis));
        output_func=outputfn,
        prob_func=probfn,
        reduction=reducefn,
        u_init=(zeros(ComplexF64, ntimes,sys.d^2,sys.d^2),
                isnothing(sys.ρ₀) ? nothing : zeros(ComplexF64, ntimes+1,sys.d,sys.d)))

    sol = ODE.solve(ensemble, ODE.RK4(), ODE.EnsembleThreads();
                    dt, saveat=dt, trajectories=length(sys),
                    batch_size=Threads.nthreads())

    sol.u[1] / length(sys), isnothing(sys.ρ₀) ? nothing : sol.u[2] / length(sys)
end



abstract type Trotter <: PLDMSolver end

function propagate_trajectory(::Type{Verlet}, sys::PLDMSystem,
                              sps0::PLDMSysPhaseSpace,
                              bps0::SolventsX.PhaseSpace,
                              dt::Real, ntimes::Integer)
    d = sys.d

    uf = sps0.uf
    vf = sps0.uf
    ub = sps0.ub
    vb = sps0.vb
    x = bps.q
    p = bps.p

    U0e = zeros(ComplexF64, ntimes,d^2,d^2)
    if !isnothing(sys.ρ₀)
        ρ = zeros(ComplexF64, ntimes+1,d,d)
        ρ[1,:,:] = (sample_weight(sys.ρ₀, uf, ub) * reconstruct_bare_ρ(uf, vb) +
                    sample_weight(sys.ρ₀, ub, uf) * reconstruct_bare_ρ(ub, vf)) / 2
        ρ₀ᵥ = Utilities.density_matrix_to_vector(sys.ρ₀)
    end

    build_dynmap!(t) = begin
        sps = PLDMSysPhaseSpace(uf, vf, ub, vb)
        bareρ = reconstruct_bare_ρ(uf, vb)
        bareρ′ = reconstruct_bare_ρ(ub, vf)
        update_dynmap!(view(U0e, t-1,:,:), bareρ, bareρ′, sys, sps0)

        if !isnothing(sys.ρ₀)
            ρ[t,:,:] = Utilities.density_matrix_vector_to_matrix(U0e[t-1,:,:] * ρ₀ᵥ)
        end
    end

    δtₓ = δt / 100
    N½ = δt / 2 / δtₓ
    # TODO: Potential candidate to generalise like done in Solvents.jl.
    propagate_xp!() = begin
        sps = PLDMSysPhaseSpace(uf, vf, ub, vb)
        for b in 1:sys.bath.nbaths
            for _ in 1:N½
                p[b] = p[b] .* 0.5 * Fbath(sys, sps, x[b], b) * δtₓ
                x[b] = x[b] .* p[b] * δtₓ
                p[b] = p[b] .* 0.5 * Fbath(sys, sps, x[b], b) * δtₓ
            end
        end
    end

    bs = sys.bath
    for t in 2:ntimes+1
        propagate_xp!()

        V = h - mapreduce((b, x) -> sum(bs.c[b] .* x) * diagm(bs.s[b]), +,
                          1:bs.nbaths, x)
        L = exp(-im * V)
        uf = L * uf
        vf = transpose(transpose(vf) * L)
        ub = conj(L) * ub
        vb = transpose(transpose(vb) * conj(L))

        propagate_xp!()

        build_dynmap_ρ!(t)
    end

    U0e, isnothing(sys.ρ₀) ? nothing : ρ
end

function propagate_trajectories(::Type{Trotter}, sys::PLDMSystem, dt::Real, ntimes::Integer;
                                output::Union{Nothing,HDF5.Group}=nothing, verbose::Bool=false,
                                kwargs...)
    U0e = zeros(ComplexF64, ntimes,sys.d^2,sys.d^2)
    isnothing(sys.ρ₀) || (ρ = zeros(ComplexF64, ntimes+1,d,d))

    outputρ = if !isnothing(output) && haskey(kwargs, :outgroup)
        Utilities.create_and_select_group(output, kwargs[:outgroup])
    else
        nothing
    end

    if !isnothing(output)
        Utilities.check_or_insert_value(output, "U0e", U0e)
        Utilities.check_or_insert_value(output, "T0e", U0e)
        Utilities.check_or_insert_value(output, "samples_done", 0)
        !isnothing(sys.ρ₀) && !isnothing(outputρ) &&
            Utilities.check_or_insert_value(outputρ, "rho", ρ)
    end

    batches = Iterators.partition(1:length(sys), Threads.nthreads())
    for samples in batches
        tasks = map(samples) do state
            ps, _ = iterate(sys, state)
            sps0, bps0 = ps
            Threads.@spawn(propagate_trajectory(Trotter, sys, sps0, bps0, dt, ntimes))
        end
        solns = fetch.(tasks)

        U0e += sum(getindex.(solns, 1))
        if !isnothing(output)
            delete_object(output, "samples_done")
            output["samples_done"] = samples[end]
            output["U0e"][:,:,:] = U0e / samples[end]
            output["T0e"][:,:,:] = TTM.get_Ts(U0e / samples[end])
            flush(output)
        end

        if !isnothing(sys.ρ₀)
            ρ += sum(getindex.(solns, 2))
            if !isnothing(outputρ)
                outputρ["rho"][:,:,] = ρ / samples[end]
                flush(outputρ)
            end
        end
        verbose && @info "Trajectories complete: $(samples[end] * 100 / length(sys))%"
    end

    U0e / length(sys), isnothing(sys.ρ₀) ? nothing : ρ / length(sys)
end



"""
    propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
              β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
              ρ0::Union{Nothing,Matrix{<:Complex}}, dt::Real,
              ntimes::Real, nmc::Integer, verbose::Bool=false,
              solver::Type{<:PLDMSolver}, sampler::Type{<:PLDMSystemSampler},
              kwargs...) where {T<:SpectralDensities.SpectralDensity}

Propagate the system using the PLDM method, employing the Bargmann
coherent states as the system basis.

Arguments:
- `Hamiltonian`: the Hamiltonian of the sub-system
- `Jw`: list of the spectral densities
- `β`: the inverse temperature of the baths
- `num_osc`: the number of discrete oscillators for each bath
- `svec`: diagonal elements of the system operators through which the
  corresponding baths interact
- `ρ0`: the initial density matrix.  If it is `nothing`, then build
  only the dynamical map
- `dt`: the time step for the propagation
- `solver`: the algorithm to use to solve the EOMs
- `sampler`: the sampling function to use for the Bargmann coherent
  states
- `nmc`: the number of Monte-Carlo samples
"""
function propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
                   β::Real, num_osc::Vector{<:Integer},
                   svec::Matrix{<:Real},
                   ρ0::Union{Nothing,Matrix{<:Complex}}, dt::Real,
                   ntimes::Real, nmc::Integer, verbose::Bool=false,
                   solver::Type{<:PLDMSolver},
                   sampler::Type{<:PLDMSystemSampler},
                   kwargs...) where {T<:SpectralDensity.SpectralDensity}
    nbaths = length(Jw)
    c = Vector{Vector{Float64}}(undef, nbaths)
    ω = Vector{Vector{Float64}}(undef, nbaths)
    s = Vector{Vector{Float64}}(undef, nbaths)

    for n in 1:nbaths
        ω[n], c[n] = SpectralDensities.discretize(Jw[n], num_osc[n])
        s[n] = svec[n,:]
    end

    bath = SolventsX.HarmonicBathX(; β, ω, c, svecs=s, nsamples=nmc)
    sys = PLDMSystem(sampler, hamiltonian=h, ρ₀=ρ0, bath, nsamples=nmc)

    propagate_trajectories(solver, sys, dt, ntimes; verbose, kwargs...)
end
