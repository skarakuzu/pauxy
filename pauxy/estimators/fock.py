from pauxy.estimators.ueg import fock_ueg
from pauxy.estimators.generic import fock_generic

def fock_matrix(system, G):
    if system.name == "UEG":
        return fock_ueg(system, G)
    elif system.name == "Generic":
        return fock_generic(system, G)
    else:
        print("# Fock matrix not implemented for {}".format(system.name))
        return None
