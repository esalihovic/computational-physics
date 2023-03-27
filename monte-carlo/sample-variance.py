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

def powerfit(x, y, xnew):
    k, m = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(m) * xnew**(k)

n = [10, 100, 1000, 10000, 100000]
N = [2, 3, 4, 5, 6]

vols = []
for i in range(5):
    arr = []
    for j in range(5):
        a = []
        for rep in range(20):
            v = Monte_carlo(N[i], n[j])
            a.append(v)
        arr.append(a)
    vols.append(arr)

var = []
for i in range(5):
    a = vols[i]
    b = []
    for j in range(5):
        x = np.var(a[j])
        b.append(x)
    var.append(b)


plt.figure(figsize=(10,6))
ys = []
for i in range(5):
    x = powerfit(n, var[i], n)
    ys.append(x)

plt.loglog(n, var[0], 'r.')
plt.plot(n, ys[0], 'r--', label="N=2")
plt.loglog(n, var[1], 'b.')
plt.plot(n, ys[1], 'b--', label="N=3")
plt.loglog(n, var[2], 'y.')
plt.plot(n, ys[2], 'y--', label="N=4")
plt.loglog(n, var[3], 'm.')
plt.plot(n, ys[3], 'm--', label="N=5")
plt.loglog(n, var[4], 'g.')
plt.plot(n, ys[4], 'g--', label="N=6")

for i in range(5):
    z = np.polyfit(np.log(n), np.log(var[i]), 1)
    p = np.poly1d(z)
    s="N=%d: y=%.6fx+(%.6f)"%(N[i], z[0], z[1])
    print(s)


plt.ylabel("sample variance")
plt.xlabel("partitions")
plt.title("Sample variance as a function of partitions")
plt.legend(bbox_to_anchor=(0.9,0.8), loc="upper right", borderaxespad=0)
plt.savefig('sample_variance.png')