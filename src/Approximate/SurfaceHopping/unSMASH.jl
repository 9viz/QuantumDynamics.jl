# All the propagation is done in the adiabatic basis but we care about
# the density matrix in the diabatic basis which can introduce a whole
# slew of notation headaches.  A consistent use of notation is
# followed throughout this module to help alleviate this pain:
# - ρ₀: initial reduced density matrix in the _diabatic_ basis
# - ω₀: initial reduced density matrix in the _adiabatic_ basis
# - ω̃ₜ: reduced density matrix in the _adiabatic_ basis at time t
# - ρ̃ₜ: reduced density matrix in the _diabatic_ basis at time t
# - ϕ: matrix of the adiabats
# - λ: current adiabatic surface
# - μ: other adiabatic surfaces
# - Greek indices: diabatic basis index
# - a, b, m, n: adiabatic basis index
# - i, j: index of current S for an adiabat
# - k: the adiabat to hop to
module unSMASH

using HDF5
using LinearAlgebra: Diagonal, eigvecs, eigen, ⋅
using ..Utilities
using ..Solvents, ..Systems, ..SpectralDensities

const references = """
- Mannouch J. R.; Richardson J. O. A mapping approach to surface hopping. J. Chem. Phys. 2023, 158, 104111.
- Lawerence J. E.; Mannouch J. R.; Richardson J. O. A size-consistent multi-state mapping apparoch to surface hopping. J. Chem. Phys. 2024, 160, 244112."""

struct unSMASHSysPhaseSpace <: Solvents.PhaseSpace
    S::AbstractMatrix{<:Real}
    ϕ::AbstractMatrix
    λ::Integer
end

struct unSMASHSys <: Systems.CompositeSystem
    h::AbstractMatrix{<:Complex}
    ρ₀::AbstractMatrix{<:Complex}
    d::Integer
    bath::Solvents.Solvent
    smats::Vector{<:AbstractMatrix}
    nsamples::Integer
end
function unSMASHSys(; Hamiltonian::AbstractMatrix{<:Complex},
                    ρ₀::AbstractMatrix{<:Complex}, bath::Solvents.Solvent,
                    nsamples::Integer)
    @assert nsamples == length(bath)
    d = size(Hamiltonian, 1)
    smats = bath.s[1] isa AbstractMatrix ? bath.s : map(Diagonal, bath.s)
    unSMASHSys(Hamiltonian, ρ₀, d, bath, smats, nsamples)
end

function sample_S(sys::unSMASHSys)
    S = zeros(3, sys.d-1)
    for i in 1:sys.d-1
        φ = 2rand()
        θ = acos(rand())
        sinφ, cosφ = sincospi(φ)
        sinθ, cosθ = sincos(θ)
        S[1,i] = sinθ * cosφ
        S[2,i] = sinθ * sinφ
        S[3,i] = cosθ
    end
    S
end

H(sys::unSMASHSys, bps::Solvents.PhaseSpace) =
    @views sys.h - mapreduce((b, x) -> sum(sys.bath.c[b] .* x) .* sys.smats[b],
                             +, 1:length(sys.bath.c), bps.q)

function Base.iterate(sys::unSMASHSys, state=1)
    state > sys.nsamples && return nothing

    map(1:sys.d) do λ
        bathps, _ = iterate(sys.bath, state)
        ϕ = eigvecs(H(sys, bathps))
        unSMASHSysPhaseSpace(sample_S(sys), ϕ, λ), bathps
    end, state+1
    # (map(λ -> unSMASHSysPhaseSpace(sample_S(sys), ϕ, λ), 1:sys.d), bathps), state+1
end
Base.length(s::unSMASHSys) = s.nsamples
Base.firstindex(s::unSMASHSys) = 1
Base.getindex(s::unSMASHSys, n::Integer) = iterate(s, n)[1]

surfidx(λ::Integer, i::Integer) = i < λ ? i : i+1
Sidx(λ::Integer, μ::Integer) = μ < λ ? μ : μ-1

function Sₘₙ(ps::unSMASHSysPhaseSpace, m::Integer, n::Integer)
    ps.λ != m && ps.λ != n && return zeros(3)

    μ = ps.λ == m ? n : m
    S = ps.S[:,Sidx(ps.λ, μ)]

    if ps.λ == n
        S[2:3] *= -1
    end

    S
end

function zₘₙ(ps::unSMASHSysPhaseSpace, m::Integer, n::Integer)
    S = Sₘₙ(ps, m, n)
    S[2] + im * S[3]
end

function sampling_weight(sys::unSMASHSys, sps0::unSMASHSysPhaseSpace, bps0::Solvents.PhaseSpace)
    gP = 0.0
    gC = 0.0

    ρP = 2prod(abs.(sps0.S[3,:]))

    ω₀ = sps0.ϕ' * sys.ρ₀ * sps0.ϕ

    for a in axes(ω₀, 2)
        if sps0.λ == a
            gP += ρP * ω₀[a,a]
            gC += 2ω₀[a,a]
        else
            z = zₘₙ(sps0, sps0.λ, a)
            # gP += z * ω₀[a,sps0.λ] + conj(z) * ω₀[sps0.λ,a]
            # gC += 1.5z * ω₀[a,sps0.λ] + 1.5conj(z) * ω₀[sps0.λ,a]
            gP += 2real(z * ω₀[a,sps0.λ])
            gC += 3real(z * ω₀[a,sps0.λ])
        end
    end

    real(gP), real(gC)
end

function build_ρ!(sys::unSMASHSys, sps::unSMASHSysPhaseSpace, gP::Real, gC::Real,
                  ρ̃::AbstractMatrix{<:Complex}, ω̃ₜ::AbstractMatrix{<:Complex})
    ω̃ₜ .= 0.0
    for a in 1:sys.d
        if a == sps.λ
            ω̃ₜ[a,a] = gP
        else
            zλa = zₘₙ(sps, sps.λ, a)
            ω̃ₜ[a,sps.λ] = gC * zλa / 2
            ω̃ₜ[sps.λ,a] = gC * conj(zλa) / 2
        end
    end

    # ω̃ₜ .*= sys.d
    ρ̃ .= sps.ϕ * ω̃ₜ * sps.ϕ'
end

function Systems.Fbath!(sys::unSMASHSys, sps::unSMASHSysPhaseSpace, f::Vector{<:AbstractVector{<:Real}})
    @inbounds for b in eachindex(sys.bath.c)
        sexp = sps.ϕ[:,sps.λ]' * sys.smats[b] * sps.ϕ[:,sps.λ]
        @. f[b] = sexp * sys.bath.c[b]
    end
end

function maintain_sign!(ϕₙ::AbstractMatrix, ϕₒ::AbstractMatrix)
    for i in axes(ϕₙ, 2)
        if real(ϕₒ[:,i]' * ϕₙ[:,i]) < 0
            ϕₙ[:,i] .*= -1
        end
    end
end

# TODO: Make this more robust.
function T!(T::AbstractMatrix, ϕₒ::AbstractMatrix, ϕₙ::AbstractMatrix, dt::Real)
    for i = axes(T, 1), j = axes(T, 2)
        T[i,j] = ϕₒ[:,i]' * ϕₙ[:,j]
    end
    T .= log(T) / dt
end

function adiabats(sys::unSMASHSys, bps::Solvents.PhaseSpace, spsₒ::unSMASHSysPhaseSpace)
    h, ϕ = eigen(H(sys, bps))
    maintain_sign!(ϕ, spsₒ.ϕ)
    h, ϕ
end

function d_λk(sys::unSMASHSys, bps::Solvents.PhaseSpace, ϕ::AbstractMatrix, E::AbstractVector, λ::Integer, k::Integer)
    d = similar.(bps.q)

    ΔE = E[k] - E[λ]
    @inbounds for b in eachindex(sys.smats)
        λsk = ϕ[:,λ]' * sys.smats[b] * ϕ[:,k] / ΔE
        @. d[b] = -sys.bath.c[b] * λsk
    end

    d
end

function propagate_trajectory(sys::unSMASHSys, sps0::unSMASHSysPhaseSpace,
                              bps0::Solvents.PhaseSpace, dt::Real, ntimes::Integer)
    ρ̃ₜ = zeros(ComplexF64, ntimes+1,sys.d,sys.d)
    ω̃ₜ = zeros(ComplexF64, ntimes+1,sys.d,sys.d)
    gP, gC = sampling_weight(sys, sps0, bps0)
    @views build_ρ!(sys, sps0, gP, gC, ρ̃ₜ[1,:,:], ω̃ₜ[1,:,:])

    T = zeros(size(sps0.ϕ))
    sps = sps0
    bps = bps0
    S = similar(sps0.S)
    λ = sps0.λ

    bs = sys.bath
    smats = map(Diagonal, bs.s)
    sexpc = similar.(bs.c)
    for t in 2:ntimes+1
        Systems.Fbath!(sys, sps, sexpc)
        _, bps = Solvents.propagate_forced_bath(bs, bps, sexpc, dt, 1)

        E, ϕₙ = adiabats(sys, bps, sps)
        T!(T, sps.ϕ, ϕₙ, dt)
        k = -1
        kidx = Integer[]
        hopped = 0
        for i in 1:sys.d-1
            μ = surfidx(λ, 1)
            ΔE = E[λ] - E[μ]
            A = [    0.0   -ΔE 2T[λ,μ];
                     ΔE    0.0 0.0;
                  -2T[λ,μ] 0.0 0.0 ]
            S[:,i] .= exp(A * dt) * sps.S[:,i]

            sign(S[3,i]) == sign(sps.S[3,i]) && continue
            if k == -1 || abs(S[3,kidx[end]]) < abs(S[3,i])
                k = μ
            end
            push!(kidx, i)
            hopped += 1
        end

        if k == -1
            sps = unSMASHSysPhaseSpace(S, ϕₙ, λ)
            @views build_ρ!(sys, sps, gP, gC, ρ̃ₜ[t,:,:], ω̃ₜ[t,:,:])
            continue
        end

        @info "Hopping from $λ to $k."
        @info "Number of spin vectors which touched equator: $hopped"

        # NOTE: We are not doing the mass-weight business since we
        # assume that the bath mode's masses are all unity.
        Δ = E[k] - E[λ]
        dλk = d_λk(sys, bps, ϕₙ, E, λ, k)
        pdotd = mapreduce(b -> bps.p[b] ⋅ dλk[b], +, 1:length(bs.c))
        ddotd = mapreduce(d -> d ⋅ d, +, dλk)
        Eₖ = 0.5 * pdotd^2 / ddotd

        if Eₖ ≥ Δ
            p = similar.(bps.p)
            pddd = pdotd / ddotd
            sqr = sqrt((Eₖ - Δ) / Eₖ)
            @inbounds for b in eachindex(bps.p)
                @. p[b] = bps.p[b] + (sqr - 1) * pddd * dλk[b]
            end
            bps = eltype(bs)(bps.q, p)
        else
            p = similar.(bps.p)
            pddd = pdotd / ddotd
            @inbounds for b in eachindex(bps.p)
                @. p[b] = bps.p[b] - 2dλk[b] * pddd
            end
            for i in kidx
                S[3,i] *= -1
            end
            bps = eltype(bs)(bps.q, p)
            sps = unSMASHSysPhaseSpace(S, ϕₙ, λ)
            @views build_ρ!(sys, sps, gP, gC, ρ̃ₜ[t,:,:], ω̃ₜ[t,:,:])
            continue
        end

        Snew = copy(S)
        for μ in 1:sys.d
            μ == k && continue

            iold = Sidx(λ, μ == λ ? k : μ)
            inew = Sidx(k, μ)

            Snew[:,inew] = S[:,iold]
            μ == λ && (Snew[2:3,inew] .*= -1)
        end
        sps = unSMASHSysPhaseSpace(Snew, ϕₙ, k)
        @views build_ρ!(sys, sps, gP, gC, ρ̃ₜ[t,:,:], ω̃ₜ[t,:,:])
    end

    ρ̃ₜ, ω̃ₜ
end

function propagate_trajectories(sys::unSMASHSys, dt::Real, ntimes::Integer;
                                output::Union{Nothing,HDF5.Group}=nothing,
                                verbose::Bool=false, kwargs...)
    ρ̃ = zeros(ComplexF64, ntimes+1,sys.d,sys.d)
    ω̃ = zeros(ComplexF64, ntimes+1,sys.d,sys.d)

    outputρ = if !isnothing(output) && haskey(kwargs, :outgroup)
        Utilities.create_and_selet_group(output, kwargs[:outgroup])
    else
        nothing
    end

    mutlock = ReentrantLock()
    ndone = 0
    nthreads = Threads.nthreads()
    # stats = @timed Threads.@threads for (sps0s, bps0) in sys
    # ρ̃ᵢ, ω̃ᵢ = mapreduce(sps0 -> propagate_trajectory(sys, sps0, bps0, dt, ntimes), .+, sps0s)
    stats = @timed Threads.@threads for ps0s in sys
        ρ̃ᵢ, ω̃ᵢ = mapreduce(ps0 -> propagate_trajectory(sys, ps0[1], ps0[2], dt, ntimes), .+, ps0s)
        # ρ̃ᵢ, ω̃ᵢ = propagate_trajectory(sys, ps0s[1][1], ps0s[1][2], dt, ntimes)
        lock(mutlock) do
            ρ̃ .+= ρ̃ᵢ
            ω̃ .+= ω̃ᵢ
            ndone += 1
            verbose && ndone % nthreads == 0 &&
                @info "Trajectories complete: $(100ndone/length(sys))%"
        end
    end
    @info "All trajectories complete\n" *
        "Time taken = $(round(stats.time; digits=3)) sec; memory allocated = $(round(stats.bytes / 1e9; digits=3)) GB; gc time = $(round(stats.gctime; digits=3)) sec"

    ρ̃ ./= length(sys)
    ω̃ ./= length(sys)
    if !isnothing(outputρ)
        outputρ["rho"] = ρ̃
        outputρ["rho_adiabatic"] = ω̃
        flush(outputρ)
    end

    ρ̃, ω̃
end

function propagate(; Hamiltonian::Matrix{<:Complex}, Jw::Vector{T},
                   β::Real, num_osc::Vector{<:Integer}, svec::Matrix{<:Real},
                   ρ0::Matrix{<:Complex}, dt::Real,
                   ntimes::Integer, nmc::Integer, verbose::Bool=false,
                   kwargs...) where {T<:SpectralDensities.SpectralDensity}
    nbaths = length(Jw)
    c = Vector{Vector{Float64}}(undef, nbaths)
    ω = Vector{Vector{Float64}}(undef, nbaths)
    s = Vector{Vector{Float64}}(undef, nbaths)

    for n in 1:nbaths
        ω[n], c[n] = SpectralDensities.discretize(Jw[n], num_osc[n])
        s[n] = svec[n,:]
    end

    bath = Solvents.HarmonicBath(; β, ω, c, svecs=s, nsamples=nmc)
    sys = unSMASHSys(; Hamiltonian, ρ₀=ρ0, bath, nsamples=nmc)

    propagate_trajectories(sys, dt, ntimes; verbose, kwargs...)
end

end
