import numpy as np
    
def vol_recursive(N):
    if N==0:
        return 1
    elif N==1:
        return 2
    else:
        return (2*np.pi/N)*vol_recursive(N-2)

v = []
for i in range(16):
    v.append(vol_recursive(i))

for i in range(16):
    print("volume of", i, "-dimensional sphere is:", v[i])