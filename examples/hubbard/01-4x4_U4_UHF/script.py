import h5py
import numpy as np
from matplotlib import pyplot as plt

filename = 'estimates.0.h5'
data = h5py.File(filename,'r')

En = np.zeros(5001)

for i in range(5001):
  number = '{0:09d}'.format(i)
  #print(number)
  #print(data['basic/energies/'+ number][:])
  En[i] = data['basic/energies/'+ number][5].real
 
average = np.sum(En[200:])/4800
print('Av : ',average)

plt.plot(En,'*')
plt.show()
#print('en: ',en[5])
