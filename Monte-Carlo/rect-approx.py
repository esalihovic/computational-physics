import numpy as np
import matplotlib.pyplot as plt


def repeat(a, c):
    r = np.stack([a for x in range(c)], axis=0)
    return r 

def RA(N, n):
    position_vector = np.linspace(0, 1, n)**2
    resulting_vector = position_vector
    for dim in range(N-1):
        resulting_vector = repeat(resulting_vector, n)
        resulting_vector = resulting_vector.T + position_vector
        print('dimension=', dim)
        print(resulting_vector)
    resulting_vector = resulting_vector.reshape(n**N)
    
    x = np.sum(np.where(resulting_vector<1, 1, 0))/n**N
    v = x*(2**N)
    return v

N = np.arange(1, 7)
n = 20
v = []
for i in N:
    vol = RA(i, n)
    v.append(vol)

plt.figure(figsize=(12, 8))
plt.plot(N, v)
plt.xlabel("dimension")
plt.ylabel("volume of a hypersphere")
plt.title("Rectangular approximation")
plt.savefig('rectangular-approx.png')