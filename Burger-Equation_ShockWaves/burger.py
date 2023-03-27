import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# defining initial conditions
N = 100 # size
c = 1.0  # wave speed
dx = 1.0/N # x step
CFL = 0.01 # Courant number 
dt = CFL*dx/c # time step
time_lim = 0.09 # time limit
n = int(time_lim/dt) # size for time

u0 = [] # initial data
u = np.zeros((n, N)) # 2D array
x = np.linspace(0, N, N)
# initial wave is sinusoidal
u0 = 3*np.sin(12.8 * x)
u[0] = u0

def LaxWendroff():
    for i in range(n-1):
        # incorporating boundary conditions
        u[i][0] = 0
        u[i][99] = 0
        for j in range(N-1):
            u[i+1][j] = u[i][j] - (CFL/4) * (u[i][j+1]**2-u[i][j-1]**2) \
                + (CFL**2/8)*((u[i][j+1] - u[i][j])*(u[i][j+1]**2-u[i][j]**2) \
                - (u[i][j] - u[i][j-1]) * (u[i][j]**2 - u[i][j-1]**2))
LaxWendroff()

# plotting
for i in range(1, n):
    plt.plot(x, u0, 'green', label='initial wave')
    plt.plot(x, u[i-1], 'pink', label = 'code-generated wave at t='\
             + str(round(dt*i, 3)))
    plt.title('formation of a shock wave for CFL = ' + str(CFL) + ' at t = '\
             + str(round(dt*i, 3)))
    # plt.legend(loc = 'lower left')
    plt.xlabel('x - direction of wave propagation')
    plt.xlim(0, 100)
    plt.ylim(-150, 150)
    plt.ylabel('wave amplitude')
    plt.pause(0.05)
    plt.show()