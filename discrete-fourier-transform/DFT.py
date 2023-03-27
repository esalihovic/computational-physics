import numpy as np

#--------------------------------TASK 1--------------------------------------
from scipy.io import wavfile
# echo sample rate - number of measurements per second
# echo data -> note that the data is a single dimensional array
samplerate, entire_data = wavfile.read('g.wav')  

#--------------------------------TASK 2--------------------------------------
# plot the entire timeseries and a short section (say, .1 sec).
# describe the functional form of the short section: is there a dominant frequency?
import matplotlib.pyplot as plt
plt.style.use("dark_background")

entire_duration = len(entire_data)/samplerate
time = np.arange(0, entire_duration, 1/samplerate) #time vector

shortSection_duration = (len(entire_data)/samplerate)/50
time2 = np.arange(0, shortSection_duration, 1/samplerate) #time vector
shortSection_data = []
for i in range(len(time2)):
    shortSection_data.append(entire_data[i])

fig, ax = plt.subplots(2)
ax[0].plot(time, entire_data, 'g')
ax[0].set_title('Entire timeseries')
ax[1].plot(time2, shortSection_data, 'g')
ax[1].set_title('Short section')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()


#-------------------------------TASK 3 - DFT-------------------------------
def DFT(arr, N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j*np.pi*k*n/N)
    
    X = np.dot(e, arr)
    return X

X = DFT(shortSection_data, len(shortSection_data))
N = len(X)
n = np.arange(N)
T = N/samplerate
freq = n/T 

plt.figure(figsize = (8, 6))
plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.title('Discrete Fourier Transform')
plt.show()

#--------------------------------TASK 4 - FFT--------------------------------
from scipy.fft import fft
y = fft(shortSection_data)
N = len(y)
n = np.arange(N)
T = N/samplerate
freq = n/T 

plt.figure(figsize = (8, 6))
plt.stem(freq, abs(y), 'r', markerfmt=" ", basefmt="-r")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.title('Fast Fourier Transform')
plt.show()

def psd(arr):
    S = []
    for el in arr:
        S.append(np.dot(el, np.conj(el)))
    return S

S = psd(y)
plt.figure(figsize = (8, 6))
plt.stem(freq, S, 'r', markerfmt=" ", basefmt="-r")
plt.xlabel('Freq (Hz)')
plt.ylabel('Power spectral density')
plt.title('Power Spectrum')
plt.show()

#--------------------------------TASK 5--------------------------------------
# use a timer to time both your version of the DFT and the python FFT
# perform both DFT and FFT and determine the scaling of computing time vs. N 
import time

SampleSize = np.array([10, 100, 1003, 9587])
data = [entire_data[0::22050], entire_data[0::2205], entire_data[0::220], \
        entire_data[0::23]]
DFT_time = []
FFT_time = []
for i in range(4):
    t1_start = time.perf_counter()
    DFT(data[i], SampleSize[i])
    t1_stop = time.perf_counter()
    DFT_time.append(t1_stop-t1_start)

    t2_start = time.perf_counter()
    fft(data[i])
    t2_stop = time.perf_counter()
    FFT_time.append(t2_stop-t2_start)
    
plt.loglog(SampleSize, DFT_time, 'purple', label='DFT')
plt.loglog(SampleSize, FFT_time, 'green', label='FFT')
plt.loglog(SampleSize, SampleSize**(1), color = "grey", label = \
           "Reference k = 1", linestyle = ":")
plt.loglog(SampleSize, SampleSize**(2), color = "grey", label = \
           "Reference k = 2", linestyle = ":")
plt.xlabel("sample size")
plt.ylabel("Computation Time")
plt.title('Scaling')
plt.legend()
plt.show()

#--------------------------------TASK 6--------------------------------------
# FFT the entire timeseries. Now, FFT back to the time domain and verify that 
# the timeseries is recovered.
from scipy.fft import ifft

y = fft(entire_data)
og = ifft(y)
N = len(og)
n = np.arange(N)
T = N/samplerate
freq = n/T 

entire_duration = len(entire_data)/samplerate
time = np.arange(0, entire_duration, 1/samplerate) #time vector

fig, ax = plt.subplots(2)
plt.rcParams["figure.figsize"] = (8, 5)
plt.ylabel('Amplitude')
plt.xlabel('Time')
ax[0].plot(time, entire_data)
ax[0].set_title('Original timeseries')
ax[1].plot(time, og)
ax[1].set_title('Back transformed timeseries')
fig.tight_layout()
plt.show()

# Remove a fraction of your frequency data by
# setting the fraction of lowest-amplitude Fourier modes to zero (as measured 
# by the power spectral density. From the remaining, non-zero amplitudes again 
# back-transform to the time domain and plot the short .1 sec section of the 
# resulting timeseries.
fa = []
for a in range(1, 11):
    fa.append(1-0.5**a)

yy = [] # stores 10 copies of S arrays
for i in range(10):
    frac = fa[i]
    arr = np.array(S)
    idx = np.argsort(arr)[:int(frac*len(arr))]
    print(i, frac, len(idx))
    print(sum(arr))
    for j in idx:
        arr[j] = 0
    yy.append(arr)

bt_yy = [] #stores 10 backtransformed arrays each of size len(S) 
for i in range(10):
    bt_array = ifft(yy[i])
    bt_yy.append(bt_array)

short_section = [[], [], [], [], [], [], [], [], [], []] 
for j in range(10):
    arr = bt_yy[j]
    for i in range(len(time2)): 
        short_section[j].append(arr[i])


for i in range(10):
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.plot(time2, short_section[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Short section' + ' $f_a =$' + str(fa[i]))
    plt.show()

"""names = ['out1.wav', 'out2.wav', 'out3.wav', 'out4.wav', 'out5.wav', \
         'out6.wav', 'out7.wav', 'out8.wav', 'out9.wav', 'out10.wav']
for i in range(10):
    wavfile.write(names[i], samplerate, bt_yy[i])  """  

#--------------------------------TASK 7-------------------------------------- 
# Use only each 100th sample in your original timeseries to perform 
# the FFT. try to recover f(t) by back-transforming. Plot the .1 sec section of 
# your timeseries to compare the data for the different cases. 
data10 = entire_data[0::10]
fft_data10 = fft(data10)
ifft_data10 = ifft(fft_data10)
time3 = np.arange(0, (len(ifft_data10)/samplerate)/50, 1/samplerate) 

data100 = entire_data[0::100]
fft_data100 = fft(data100)
ifft_data100 = ifft(fft_data100)
time4 = np.arange(0, (len(ifft_data100)/samplerate)/50, 1/samplerate) 

shortSection10 = []
shortSection100 = []
for i in range(len(time3)):
    shortSection10.append(ifft_data10[i])
for i in range(len(time4)):
    shortSection100.append(ifft_data100[i])

fig, ax = plt.subplots(3)
plt.rcParams["figure.figsize"] = (8, 5)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
ax[0].plot(time2, shortSection_data)
ax[0].set_title('original short section')
ax[1].plot(time3, shortSection10)
ax[1].set_title('back-transformed short section, N=10')
ax[2].plot(time4, shortSection100)
ax[2].set_title('back-transformed short section, N=100')
fig.tight_layout()
plt.show()
