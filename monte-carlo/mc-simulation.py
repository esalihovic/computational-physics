import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
    
def dist(l):
    l_squared = []
    for el in l:
        l_squared.append(el**2)
    return np.sqrt(sum(l_squared))

def Monte_carlo(N, n):
    counter = 0 #number of "Hits", aka number of points which lie inside the sphere
    for i in range(n):
        # generates random numbers uniformly distributed between zero and one
        x = np.random.uniform(0, 1, N)
        R = dist(x) # Calculate distance from the orgin
        if R <= 1: # Check for a hit
            counter += 1 # If there's a hit increment Zn
    vol = 2**N*counter/n
    return vol

# computing volumes of hyperspheres for dimensions 1 to 30
N = np.arange(1, 16)
n = 1000
v = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in range(15):
    for j in range(100):
        vol = Monte_carlo(i, n)
        v[i].append(vol)
# averaging 100 obtained volumes
v_avg = []
for i in range(15):
    vi_avg = stats.mean(v[i])
    v_avg.append(vi_avg)
    
# plotting
plt.figure(figsize=(12, 8))
plt.plot(N, v_avg)
plt.xlabel("dimension")
plt.ylabel("volume of a hypersphere")
plt.title("Monte Carlo")
plt.savefig('monte-carlo.png')

# statistics
std = []
prefactor = []
stdError = []

for i in range(15):
    s = stats.stdev(v[i])
    std.append(s)
    m = stats.mean(v[i])
    prefactor.append(m)
    e = s/np.sqrt(100)
    stdError.append(e)
    
for i in range(15):
    print('The prefactor for the volume of a ', i+1, "-dimensional sphere is" \
          , prefactor[i], '\u00B1', stdError[i])