from juliacall import Main as jl
import numpy as np 

jl.include("./hubbard_dmrg.jl")
jl.seval("using ITensorGaussianMPS")
class interface:
  def __init__(self, nx, ny, nup, ndwn, t, U, xperiodic, yperiodic):
  #def __init__(self):
    #self.nx = nx
    # System size
    self.nx = nx
    self.ny = ny
    self.nf_up = nup
    self.nf_dwn = ndwn
    self.n = self.nx * self.ny

    self.U = U
    self.t = t

    # Half filling
    # Other fillings don't work right now,
    # need to fix `hubbard_dmrg.jl`!
    self.nf = self.n
    self.nf_up = self.n // 2
    self.nf_dn = self.n - self.nf_up

    # Boundary conditions in the y-direction
    self.yperiodic = yperiodic

    # DMRG run parameters
    dmrg_nsweeps = 10
    dmrg_cutoff = 1e-6
    dmrg_maxdim = 100

    os_up, os_dn, _ = jl.hubbard(nx=self.nx, ny=self.ny, U=self.U, t=self.t, yperiodic=self.yperiodic)
    self.h_up = jl.hopping_hamiltonian(os_up)
    self.h_dn = jl.hopping_hamiltonian(os_dn)


    self.H, self.psi_trial = jl.hubbard_dmrg(nx=self.nx, ny=self.ny, nf_up=self.nf_up, nf_dn=self.nf_dn, U=self.U, t=self.t, yperiodic=self.yperiodic, nsweeps=dmrg_nsweeps, maxdim=dmrg_maxdim, cutoff=dmrg_cutoff)
    self.psi_walker = self.psi_trial

  def calc_overlap(self):

    # Form the Slater Determinant.
    # This would be a walker coming from AFQMC.
    self.overlap =  jl.inner(self.psi_trial, self.psi_walker)
    return self.overlap
    
  def calc_log_overlap(self,psi_walker):    
    #print(jl.loginner(self.psi_trial, self.psi_walker))
    return jl.loginner(self.psi_trial, psi_walker)

  def calc_energy_trial(self):
    print('Calculating energy trial')
    self.overlap =  jl.inner(self.psi_trial, self.psi_walker)
    energy = jl.inner(jl.prime(self.psi_trial), self.H, self.psi_trial)
    print('energy is: ',energy)
    return energy/self.overlap, 0.0, 0.0
    
  def calc_energy(self):
    print('Calculating energy')
    energy = jl.inner(jl.prime(self.psi_trial), self.H, self.psi_walker)
    print('energy is: ',energy)
    return energy, 0.0, 0.0

  def calc_energy_walker_H_walker(self):
    #print(jl.inner(jl.prime(self.psi_walker), self.H, self.psi_walker))
    return jl.inner(jl.prime(self.psi_walker), self.H, self.psi_walker)

  def convert_to_mps(self,psi_walker):
    print('Converting to MPS')
    phi_up = psi_walker[:,:self.nf_up]
    phi_dn = psi_walker[:,self.nf_up:]
    eigval_cutoff = 1e-6
    cutoff = 1e-6
    maxdim = 100
    #print('phi_up', phi_up)
    #print('phi_dn', phi_dn)
    #print('site_indices',jl.siteinds(self.psi_trial))
    self.psi_walker = jl.slater_determinant_to_mps(jl.siteinds(self.psi_trial), phi_up, phi_dn, eigval_cutoff=eigval_cutoff, cutoff=cutoff, maxdim=maxdim)


  def calc_local_density_overlap(self):
    self.n, self.overlap = jl.local_density(self.psi_trial, self.psi_walker)
    self.nup = self.n[0]
    self.ndn = self.n[1]
    return np.array(self.nup)/self.overlap, np.array(self.ndn)/self.overlap
#if __name__ == '__main__':
#  obj = interface()

#  overlap = obj.calc_overlap()
#  print('< psi_trial | psi_walker > = ',overlap)

#  log_overlap = obj.calc_log_overlap()
#  print('log(< psi_trial | psi_walker > = )',log_overlap)

#  en = obj.calc_energy_trial_H_walker()
#  print('< psi_trial | H | psi_walker > = ', en)

#  en_2 = obj.calc_energy_walker_H_walker()
#  print('< psi_walker | H | psi_walker > = ', en_2)
