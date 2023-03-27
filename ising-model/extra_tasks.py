from random import seed
from random import randint
import numpy as np 
import math
import matplotlib.pyplot as plt

def generate_state(size):
    x = int(np.sqrt(size))
    return np.random.choice([1, -1], size=(x, x))
def energy(lattice, N):
    L = int(np.sqrt(N))
    energy = 0
    for row in range(L):
        for col in range(L):
            site = lattice[row][col]
            
            if row == 0 or row == L-1 and col == 0 or col == L-1: #corners
                nb = lattice[row][(col+L-1)%L]+lattice[(row+L-1)%L][col]
                energy += site*nb
                
            elif row == 0 or row == L-1 and col != 0 or col != L-1:
                nb=lattice[row][(col+L-1)%L]+lattice[(row+L-1)%L][col] \
                    + lattice[row][(col + L +1) % L]
                energy += site*nb
            
            elif col == 0 or col == L-1 and row != 0 or row != L-1:
                nb=lattice[(row+L-1)%L][col]+lattice[row][(col+L-1)%L] + \
                    lattice[(row+L+1)%L][col]
                energy += site*nb
            else:
                nb=lattice[(row+L-1)%L][col] + lattice[row][(col+L+1)%L] \
                  + lattice[(row+L+1)%L][col] + lattice[row][(col+L-1)%L]
                energy += site*nb
    return (-1)*energy

#redefining monte carlo using different seeds 
def new_monte_carlo(lattice, N, oldE, T):
    L = int(np.sqrt(N))
    
    # using different seeds between 0-10 to initialize the RNG
    seed(randint(0, 10))
    row = randint(0, L-1)
    col = randint(0, L-1)
    
    #site = lattice[row, col]
    lattice[row, col] *= -1
    
    E = energy(lattice, N)
    dE = E - oldE
    r = math.exp(-dE/T)
    
    # energy of the system would be lowered by reversing the spin, so the spin
    # is flipped and the system moves into a different microstate
    if dE <= 0:
        return lattice, E
    
    else:
        if np.random.rand() < r:
            return lattice, E
        else:
            lattice[row, col] *= -1
            return lattice, oldE

def metropolis(lattice, N, T):
    lattice_energy1 = energy(lattice, N) 
    lattice_energy = [lattice_energy1]
    for i in range(1, N+1):
        state, E = new_monte_carlo(lattice, N, lattice_energy[i-1], T)
        lattice_energy.append(E)
    return state
def magnetization(lattice, N):
    return abs(np.sum(lattice)/N)
def ising(n, state, N, T):
    E = [] # energies
    M = [] # magnetization per site
    for i in range(n0):
        updated_state = metropolis(state, N, T)
        E.append(energy(updated_state, N))
        M.append(magnetization(updated_state, N))
    return E, M

L1 = 20
N1 = L1**2
k_B = J = 1
initial_state = generate_state(N1)
T_prime = 0.1

n0 = 100
Es = [] # stores 10 arrays each of n0 elements
Ms = []
# repeat the simulation several times for each observable
for i in range(10):
    print(i)
    e, m = ising(n0, initial_state, N1, T_prime)
    Es.append(e)
    Ms.append(m)

new_Es = [] # should contain n0 arrays of 10 elements
new_Ms = []
for i in range(n0):
    new_Es.append([])
    new_Ms.append([])
for i in range(10):
    array1 = Es[i]
    array2 = Ms[i]
    for j in range(n0):
        new_Es[j].append(array1[i])
        new_Ms[j].append(array2[i])

errs_E = [] # single array of n0 elements
for array in new_Es:
    e = np.std(array)/np.sqrt(10)
    errs_E.append(e)
errs_M = [] # single array of n0 elements
for array in new_Ms:
    m = np.std(array)/np.sqrt(10)
    errs_M.append(m)
    
n = np.linspace(0, n0, n0)
plt.plot(n, errs_E,  color="purple", linewidth=0.9)
plt.xlabel('iterations')
plt.xticks(np.arange(0, n0+1, 10))
plt.ylabel('Standard error')
plt.title('Standard errors of energy as a function of iterations')
plt.show()
plt.plot(n, errs_M, color='green', linewidth=0.9)
plt.xlabel('iterations')
plt.xticks(np.arange(0, n0+1, 10))
plt.ylabel('Standard error')
plt.title('Standard errors of magnetization as afunction of iterations')
plt.show()


#--------------------------now for temperature-------------------------------
T = [0.1, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 10.0]
new_Es1 = [[],[],[],[],[],[],[],[],[]]
new_Ms1 = [[],[],[],[],[],[],[],[],[]]

n_min = 70
for x, temp in enumerate(T):
    for i in range(n_min):
        print(x, temp, i)
        updated_state = metropolis(initial_state, N1, temp)
        new_Es1[x].append(energy(updated_state, N1))
        new_Ms1[x].append(magnetization(updated_state, N1))
    
errors_E2 = []
for array in new_Es1:
    x = np.std(array)/np.sqrt(len(array))
    errors_E2.append(x)
errors_M2 = []
for array in new_Ms1:
    y = np.std(array)/np.sqrt(len(array))
    errors_M2.append(y)

plt.plot(T, errors_E2,  color="blue", linewidth=0.9)
plt.xlabel('Temperatures')
plt.xticks(np.linspace(0, 10, 11))
plt.ylabel('Standard error')
plt.title('Standard error of system energy as a function of temperature')
plt.show()
plt.plot(T, errors_M2, color='orange', label = 'Numerical', linewidth=0.9)
plt.xlabel('Temperature')
plt.xticks(np.linspace(0, 10, 11))
plt.ylabel('Standard error')
plt.title('Standard error od system energy as a function of temperature')
plt.show()

#--------------------------TASK 10-------------------------------
L2 = 50
N2 = L1**2
initial_state2 = generate_state(N2)
T_prime2 = 0.1

n0 = 100
Es = [] # stores 10 arrays each of n0 elements
Ms = []
for i in range(10):
    e, m = ising(n0, initial_state, N2, T_prime2)
    Es.append(e)
    Ms.append(m)
E_fin = (np.array(Es[0])+np.array(Es[1])+np.array(Es[2])+np.array(Es[3])\
         +np.array(Es[4])+np.array(Es[5])+np.array(Es[6])+np.array(Es[7])\
             +np.array(Es[8])+np.array(Es[9]))/10 
M_fin = (np.array(Ms[0])+np.array(Ms[1])+np.array(Ms[2])+np.array(Ms[3])\
         +np.array(Ms[4])+np.array(Ms[5])+np.array(Ms[6])+np.array(Ms[7])\
             +np.array(Ms[8])+np.array(Ms[9]))/10 

# from this plot we will observe n_min
n = np.linspace(0, n0, n0)
plt.plot(n, E_fin,  color="purple", linewidth=2)
plt.xlabel('time [arbitrary units]')
plt.xticks(np.arange(0, n0+1, 10))
plt.ylabel('Energy of a system')
plt.title('Timeseries of energies')
plt.show()
plt.plot(n, M_fin, color='green', linewidth=0.9)
plt.xlabel('time [arbitrary units]')
plt.xticks(np.arange(0, n0+1, 10))
plt.ylabel('Magnetization per site')
plt.title('Time series of magnetization per site')
plt.show()

new_Es = [] # should contain n0 arrays of 10 elements
new_Ms = []
for i in range(n0):
    new_Es.append([])
    new_Ms.append([])
for i in range(10):
    array1 = Es[i]
    array2 = Ms[i]
    for j in range(n0):
        new_Es[j].append(array1[i])
        new_Ms[j].append(array2[i])

errs_E = [] # single array of n0 elements
for array in new_Es:
    e = np.std(array)/np.sqrt(10)
    errs_E.append(e)
errs_M = [] # single array of n0 elements
for array in new_Ms:
    m = np.std(array)/np.sqrt(10)
    errs_M.append(m)
    
n = np.linspace(0, n0, n0)
plt.plot(n, errs_E,  color="purple", linewidth=0.9)
plt.xlabel('iterations')
plt.xticks(np.arange(0, n0+1, 10))
plt.ylabel('Standard error')
plt.title('Standard errors of energy as a function of iterations')
plt.show()
plt.plot(n, errs_M, color='green', linewidth=0.9)
plt.xlabel('iterations')
plt.xticks(np.arange(0, n0+1, 10))
plt.ylabel('Standard error')
plt.title('Standard errors of magnetization as afunction of iterations')
plt.show()


#--------------------------now for temperature-------------------------------
T = [0.1, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 10.0]
new_Es1 = [[],[],[],[],[],[],[],[],[]]
new_Ms1 = [[],[],[],[],[],[],[],[],[]]

n_min = 70
for x, temp in enumerate(T):
    for i in range(n_min):
        print(x, temp, i)
        updated_state = metropolis(initial_state, N1, temp)
        new_Es1[x].append(energy(updated_state, N1))
        new_Ms1[x].append(magnetization(updated_state, N1))
    
errors_E2 = []
for array in new_Es1:
    x = np.std(array)/np.sqrt(len(array))
    errors_E2.append(x)
errors_M2 = []
for array in new_Ms1:
    y = np.std(array)/np.sqrt(len(array))
    errors_M2.append(y)

plt.plot(T, errors_E2,  color="blue", linewidth=0.9)
plt.xlabel('Temperatures')
plt.xticks(np.linspace(0, 10, 11))
plt.ylabel('Standard error')
plt.title('Standard error of system energy as a function of temperature')
plt.show()
plt.plot(T, errors_M2, color='orange', label = 'Numerical', linewidth=0.9)
plt.xlabel('Temperature')
plt.xticks(np.linspace(0, 10, 11))
plt.ylabel('Standard error')
plt.title('Standard error od system energy as a function of temperature')
plt.show()