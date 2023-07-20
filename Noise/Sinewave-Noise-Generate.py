import numpy as np
from matplotlib import pyplot as plt
import math 

#Creating a sine wave 
freq=50
time_period=1/freq #20ms
time = time_period*2 #40ms (Time period)
amplitude=2

t=np.linspace(0, time, 1000, endpoint=True)
 #(start, end, total number of points from start will be spaced, whether endpoint will be included or not)
x=5*math.pi*freq*t #sine angle
# y = asin(omega * t)

sample_signal=amplitude*np.sin(x)+amplitude*np.sin(5*x)
noise = np.random.normal(2,1,1000)

noiseSignal = sample_signal*noise

# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise


plt.subplot(3, 1, 1)
plt.title('Sample Signal')
plt.plot(t, sample_signal,'tab:orange')

plt.subplot(3, 1, 2)
plt.title('Random Noise')
plt.plot(t, noise,'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Signal combined with noise')
plt.plot(t, noiseSignal)
plt.tight_layout()


plt.show()