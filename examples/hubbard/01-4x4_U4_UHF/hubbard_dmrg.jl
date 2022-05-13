using ITensors

using ITensors: site, coef, name

import ITensors: inner, expect

# o = ∑ⱼ₌₁ⁿ αⱼ oⱼ
function inner(ϕ::MPS, o::OpSum, ψ::MPS)
  s = siteinds(ψ)
  n = length(ψ)
  ϕᴴ = sim(linkinds, dag(ϕ))

  # Cache the environments
  E = Dict()
  E[1:0] = ITensor(1.0)
  E[(n + 1):n] = ITensor(1.0)
  for j in 1:n
    E[1:j] = E[1:(j - 1)] * ϕᴴ[j] * ψ[j]
  end
  for j in reverse(1:n)
    E[j:n] = E[(j + 1):n] * ϕᴴ[j] * ψ[j]
  end
  # Compute the overlap
  
  ϕψ = (E[1:1] * E[2:n])[]

  # Compute the expectation values
  ϕoψ = Dict{String,Dict{Int,Number}}()
  for tⱼ in o
    αⱼ = coef(tⱼ)
    if isreal(αⱼ)
      αⱼ = real(αⱼ)
    end
    oⱼ = only(ops(tⱼ))
    o = name(oⱼ)
    j = site(oⱼ)
    Oⱼ = αⱼ * op(o, s, j)
    Oψⱼ = noprime(Oⱼ * ψ[j])
    get!(ϕoψ, o, Dict{Int,Number}())
    ϕoψ[o][j] = (E[1:(j - 1)] * ϕᴴ[j] * E[(j + 1):n] * Oψⱼ)[]
  end
  return ϕoψ, ϕψ
end

function inner_per_site(ϕ::MPS, ops::Vector{String}, ψ::MPS)
  n = length(ψ)
  opsum = OpSum()
  for o in ops
    for j in 1:n
      opsum += o, j
    end
  end
  ϕoψ, ϕψ = inner(ϕ, opsum, ψ)
  inner_ops = [[ϕoψ[o][j] for j in 1:n] for o in ops]
  return inner_ops, ϕψ
end

function local_density(ϕ::MPS, ψ::MPS)
  return inner_per_site(ϕ, ["Nup", "Ndn"], ψ)
end


function hubbard(; nx, ny, U, t, yperiodic=false)
  lattice = square_lattice(nx, ny; yperiodic)
  h_up = OpSum()
  for b in lattice
    h_up += -t, "Cdagup", b.s1, "Cup", b.s2
    h_up += -t, "Cdagup", b.s2, "Cup", b.s1
  end
  h_dn = OpSum()
  for b in lattice
      h_dn += -t, "Cdagdn", b.s1, "Cdn", b.s2
      h_dn += -t, "Cdagdn", b.s2, "Cdn", b.s1
  end
  h_interacting = h_up + h_dn
  for j in 1:(nx * ny)
    h_interacting += U, "Nupdn", j
  end
  return h_up, h_dn, h_interacting
end

function hubbard_dmrg(; nx, ny, nf_up=((nx * ny) ÷ 2), nf_dn=(nx * ny - nf_up), U, t, yperiodic=false, nsweeps, maxdim, cutoff)
  n = nx * ny
  nf = nf_up + nf_dn

  # Starting state with the specified filling
  # TODO: This needs to be modified to handle general
  # filling.

  state(j) = j ≤ nf ? (isodd(j) ? "↑" : "↓") : "0"

  # Hilbert space of the physical sites
  sites = siteinds("Electron", n; conserve_qns=true)

  # Hamiltonian
  h_up, h_dn, h_interacting = hubbard(; nx, ny, U, t, yperiodic)
  h = h_up + h_dn + h_interacting
  H = MPO(h, sites)

  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  psi0 = randomMPS(sites, state; linkdims=10)

  # Get the trial wavefunction `psi_trial`
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
  return H, psi
end







