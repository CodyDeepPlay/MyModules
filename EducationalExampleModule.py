# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:25:24 2018

@author: Mingming
"""
import numpy as np
import matplotlib.pyplot as plt 


#%%
################################################################
##################  EDUCATIONAL EXAMPLE ########################
################################################################

"""
FFT EXAMPLE
https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python

An educational example showing how to use fft to analyze signal in 
frequency domain.
"""
def fftExample():
    from scipy.fftpack import fft
    # Number of samplepoints
    N = 600
    # sample spacing
    T = 1.0 / 800.0
    # Signal in time domain
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    
    # Conduct fft and plot the signal as show case
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x, y)
    plt.xlabel('time in s')
    plt.subplot(2,1,2)
    plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
    plt.xlabel('frequency')
    plt.grid()
    plt.show()







#%%  how to run a piece of code and check it is running time

import timeit
        
SETUP_CODE = ''' 
import EMGModule
import numpy as np
detect_start = 24
detect_end   = 96
my_wave = np.random.randint( -100 , high= 100, size = 257 )
'''
      
TEST_CODE = ''' 
threshold_res_label = EMGModule.TimeDomainTh(my_wave, detect_start, detect_end, care_noise='yes')  '''

# timeit.repeat statement 
times = timeit.repeat(setup = SETUP_CODE, 
                          stmt = TEST_CODE, 
                          repeat = 3, 
                          number = 100) 
             
print('NormWave processing time: {}'.format(np.mean(times)))
        




#%%  how to plot and display time series signal in real time

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

plt.ion()  # turn the interactive mode on

fig = plt.figure(1)  # an instance of a figure plotting object
plt.clf()            # clear anything on current figure
# add an Axes instance to the figure as part of a subplot arrangement
ax     = fig.add_subplot(111) # Either a 3-digit integer or three separate integers describing the position of the subplot
line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(x + phase)) # update the ydata for next frame display
    fig.canvas.draw()                  # update the figure that has been altered
    fig.canvas.flush_events()    # speed up matplotlib plotting times

    
#%%  how to take the input from key board typing

from pynput.keyboard import Key, Listener

def on_press(key):
    print('{0} pressed'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

def on_release(key):
    #string = ('{0} release'.format(
      #  key))
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()


#%%  how to plot and display signals real time

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

plt.ion()  # turn the interactive mode on

fig = plt.figure(1)  # an instance of a figure plotting object
plt.clf()            # clear anything on current figure
# add an Axes instance to the figure as part of a subplot arrangement
ax     = fig.add_subplot(111) # Either a 3-digit integer or three separate integers describing the position of the subplot
line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 100):
    
    line1.set_ydata(np.sin(x + phase)) # update the ydata for next frame display
    fig.canvas.draw()                  # update the figure that has been altered
    fig.canvas.flush_events()    # speed up matplotlib plotting times



