import numpy as np
import random 
import math
import matplotlib.pyplot as plt

#--------------------------------TASK 2-----------------------------------
# function which generates a random state/lattice/spin configuration
def generate_state(size):
    x = int(np.sqrt(size))
    return np.random.choice([1, -1], size=(x, x))
k_B = J = 1

#--------------------------------TASK 3-----------------------------------
T_prime = 0.1
L = 20
N = L**2
initial_state = generate_state(N)
plt.imshow(initial_state)
plt.xticks(np.arange(0, L+1, 10))
plt.yticks(np.arange(0, L+1, 10))
plt.title("Initial state")
plt.show()
# function which computes total energy of a given state/lattice/spin configuration
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

#--------------------------------TASK 4-----------------------------------
# function that flips a randomly selected spin based on the energy difference
# calculated
def monte_carlo(lattice, N, oldE, T):
    L = int(np.sqrt(N))
    
    row = random.randint(0, L-1)
    col = random.randint(0, L-1)
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

#--------------------------------TASK 5-----------------------------------
# Code a function that produces N attempted site updates, hence, each site is 
# considered once on average (due to randomness, some sites will not be updated 
# but others multiple times
def metropolis(lattice, N, T):
    lattice_energy1 = energy(lattice, N) 
    lattice_energy = [lattice_energy1]
    for i in range(1, N+1):
        state, E = monte_carlo(lattice, N, lattice_energy[i-1], T)
        lattice_energy.append(E)
    return state

new_state = metropolis(initial_state, N, T_prime) # carrying out one system update

#--------------------------------TASK 6-----------------------------------
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

n0 = 100
Es = [] # stores 10 arrays each of n0 elements
Ms = []
for i in range(10):
    e, m = ising(n0, initial_state, N, T_prime)
    Es.append(e)
    Ms.append(m)

print(Es)
print(Ms)

# averaging (could be written better with a loop probably but i got some errors
# which i didnt know how to fix when i tried to do so)
E_fin = (np.array(Es[0])+np.array(Es[1])+np.array(Es[2])+np.array(Es[3])\
         +np.array(Es[4])+np.array(Es[5])+np.array(Es[6])+np.array(Es[7])\
             +np.array(Es[8])+np.array(Es[9]))/10 
M_fin = (np.array(Ms[0])+np.array(Ms[1])+np.array(Ms[2])+np.array(Ms[3])\
         +np.array(Ms[4])+np.array(Ms[5])+np.array(Ms[6])+np.array(Ms[7])\
             +np.array(Ms[8])+np.array(Ms[9]))/10 

print(E_fin)
print(M_fin)

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

#--------------------------------TASK 7-----------------------------------
T = [0.1, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 10.0]
new_Es = [[],[],[],[],[],[],[],[],[]]
new_Ms = [[],[],[],[],[],[],[],[],[]]
    
for x, temp in enumerate(T):
    for i in range(70):
        print(x, temp, i)
        updated_state = metropolis(initial_state, N, temp)
        new_Es[x].append(energy(updated_state, N))
        new_Ms[x].append(magnetization(updated_state, N))
    plt.imshow(updated_state)
    plt.title("Final state at {}".format(temp))
    plt.show()

print(len(new_Es))
print(len(new_Ms))

new_E_fin = []
for array in new_Es:
    new_E_fin.append(np.mean(array))
new_M_fin = []
for array in new_Ms:
    a = np.mean(array)
    print(a)
    new_M_fin.append(a)

print(len(new_E_fin))
print(len(new_M_fin))

#--------------------------------TASK 8-----------------------------------
# theoretical value
T_crit = 2/(np.log(1+2**(1/2)))
sol = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(9):
    if  (T[i] - T_crit) < 0:
        sol[i] = (1-(math.sinh(2/T[i]))**(-4))**(1/8)
    elif (T[i] - T_crit) >= 0:
        sol[i] = 0

plt.plot(T, new_E_fin,  color="blue", linewidth=0.9)
plt.xlabel('Temperatures')
plt.xticks(np.linspace(0, 10, 11))
plt.ylabel('Energy of a system')
plt.title('Energy as a function of temperature')
plt.show()
plt.plot(T, new_M_fin, color='orange', label = 'Numerical', linewidth=0.9)
plt.plot(T, sol, color='red', label='Theoretical',linewidth=0.9)
plt.xlabel('Temperature')
plt.xticks(np.linspace(0, 10, 11))
plt.ylabel('Magnetization per site')
plt.title('Magnetization per site as a function of temperature')
plt.show()