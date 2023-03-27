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
    resulting_vector = resulting_vector.reshape(n**N)
    
    x = np.sum(np.where(resulting_vector<1, 1, 0))/n**N
    v = x*(2**N)
    return v

def powerfit(x, y, xnew):
    k, m = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(m) * xnew**(k)

n = [5, 10, 15, 20, 25]
N = [2, 3, 4, 5, 6]

# array of arrays
vol = []
for i in range(5):
    arr = []
    for j in range(5):
        v = RA(N[i], n[j])
        arr.append(v)
    vol.append(arr)

exact = [3.14159265358979, 4.18879020478639, 4.93480220054468, \
         5.26378901391432, 5.16771278004997]
err = []
for i in range(5):
    arr = vol[i]
    e = []
    for j in range(5):
        diff = np.abs(arr[j]-exact[i])
        e.append(diff)
    err.append(e)


plt.figure(figsize=(10,6))
ys = []
for i in range(5):
    x = powerfit(n, err[i], n)
    ys.append(x)

plt.loglog(n, err[0], 'r.')
plt.plot(n, ys[0], 'r--', label="N=2")
plt.loglog(n, err[1], 'b.')
plt.plot(n, ys[1], 'b--', label="N=3")
plt.loglog(n, err[2], 'y.')
plt.plot(n, ys[2], 'y--', label="N=4")
plt.loglog(n, err[3], 'm.')
plt.plot(n, ys[3], 'm--', label="N=5")
plt.loglog(n, err[4], 'g.')
plt.plot(n, ys[4], 'g--', label="N=6")

for i in range(5):
    z = np.polyfit(np.log(n), np.log(err[i]), 1)
    p = np.poly1d(z)
    s="N=%d: y=%.6fx+(%.6f)"%(N[i], z[0], z[1])
    print(s)

plt.ylabel("error")
plt.xlabel("number of integration points, n")
plt.title("Error as a function of number of integration points")
plt.legend(bbox_to_anchor=(0.85,0.92), loc="upper left", borderaxespad=0)
plt.savefig('error-vs-num_of_integration_pts.png')