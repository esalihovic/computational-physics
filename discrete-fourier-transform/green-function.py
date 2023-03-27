import numpy as np
import matplotlib.pyplot as plt

beta = [0.05, 1.5]

alpha = 1

w = np.linspace(0, 20, 1000)
G_w1 = []
G_w2 = []
for el in w:
    G_w1.append((-1)*0.5**np.pi/(el**2 + 2j*beta[0]*el - alpha**2))
    G_w2.append((-1)*0.5**np.pi/(el**2 + 2j*beta[1]*el - alpha**2))

# applying FFT
from scipy.fft import fft
y1 = fft(G_w1)
y2 = fft(G_w2)

fig, ax = plt.subplots(2)
ax[0].plot(w, np.real(y1), 'b')
ax[0].set_title('Forced Harmonic Oscillator with β = 0.05')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Amplitude')

ax[1].plot(w, np.real(y2), 'b')
ax[1].set_title('Forced Harmonic Oscillator, β=1.5')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Amplitude')

plt.tight_layout()