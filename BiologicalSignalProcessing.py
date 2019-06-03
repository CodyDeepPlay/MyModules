# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:47:22 2018

@author: Mingming

This file is the customized module to analyze biological signal process
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from six.moves import cPickle as pickle
import pandas as pd
from scipy import signal
from detect_peaks import detect_peaks  # a customized function downloaded online
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures


import math

# BELOW ARE CUSTOMIZED FUNCTIONS
##########################################################################################################

"""
An old version of artifac_loc()
artifac_loc_old(), identify the artifact location
INPUT:
singal,    a channel of input signal, a 1-D numpy array
threshold, the threshold one defines to detect the negative going spikes in the signal,
           here the defined threshold value will be based on the user's empirical experience 
           from his data or input signal       
OUTPUT:
artifact_end, the index of the raw signal, to indicate when the artifact ends, assuming artifact will last for a short
              period of time.      
"""
def artifac_loc_old(signal, threshold):
    # in the raw signal, find out those index where the values are below the threshold
    artifact_loc       = np.squeeze(np.where(signal<threshold))     # index, where values are below the threshold, detected artifact last a duration of time
    artifact_loc_diff  = np.diff(artifact_loc)                      # calculate the difference of the location index
    artifact_end_trans = np.squeeze(np.where(artifact_loc_diff>50)) # identify where the big gaps are, to judge the end of a artifact

    num_detect_spikes  = len(artifact_loc[artifact_end_trans]) # the detected artifacts, indexes here are the ends of the artifact duration
    total_num          = num_detect_spikes + 1                 # add the last artifact in the artifact_loc, cause diff() will miss the last one
    artifact_end       = np.zeros(total_num,dtype=int)         # prelocate the space for this parameter
    artifact_end[0:-1] = artifact_loc[artifact_end_trans]
    artifact_end[-1]   = artifact_loc[-1]                      # add the last artifact index duration end
   
    # some of the negative crossing end might be the evoked response, instead of artifact
    # the following code is doing the double check it is artifact, using the lowest amplitude 
    # within the investigation window
    
    # fs = 4k, let's define a time window, include 40 data points before
    # and 20 data points after each "artifact end" point
    real_arti_loc = np.zeros(len(artifact_end), dtype = int)
    for arti_id in range(len(artifact_end)):
        head = artifact_end[arti_id]-40
        tail = artifact_end[arti_id]+20
        signal_section = signal[head : tail]           # take current "artifact end" and move 40 point before
        real_index = np.squeeze(np.where(signal_section == min(signal_section)))     # using the lowest point as the real artifact location
        real_arti_loc[arti_id] = head + real_index     # the real index in the original time series 

    return real_arti_loc




"""
An updated version of artifact_loc(), to detect the artifact of the stimulation pulses, using an online customized function
detect_peaks, like the findpeak() in Matlab

INPUT:
mysingal,  a channel of input signal, a 1-D numpy array
threshold, the threshold one defines to detect the negative going spikes in the signal,
           here the defined threshold value will be based on the user's empirical experience 
           from his data or input signal       
method,    what method to use to determine the stimulation artifact given a time window of signal
           method = 0, default option, using first negative peak method
           method = 1, using the minimum value as the indicator of the stim artifact           
           
spike_distance, an interger, in time points
                minimum spike distance the user wants to set,
                this is used to distinguish the different stimulation artifact.
                i.e. default value spike_distance=500, means the minimum distance between two artifacts
                is expected to be larger than 500 data points. If fs = 8k, 500 data points means 62.5 ms.
half_stim_window, an integer, in time points
                how long is a half stimulation time window, the default value is 50 data points.
           
OUTPUT:
real_arti_loc, indicate where the artifact locates, based on the first negative peak that was detected
artifact_end, the index of the raw signal, to indicate when the artifact ends, assuming artifact will last for a short
              period of time

"""
def artifac_loc(mysignal, threshold, method=0, spike_distance=350, half_stim_window=50):           
    # in the raw signal, find out those index where the values are below the threshold
    # only using negative going artifact to identify the signal section where it might has artifact
    
    # first use 30 times of the given threshold to determine the stimulation artifact
    artifact_loc       = np.squeeze(np.where(mysignal<5*threshold))     # index, where values are below the threshold, detected artifact last a duration of time
    artifact_loc_diff  = np.diff(artifact_loc)                        # calculate the difference of the location index
    artifact_end_trans = np.squeeze(np.where(artifact_loc_diff>spike_distance))  # identify where the big gaps are, to judge the end of a artifact

    # roughly find the beginning of each detect artifact
    num_detect_spikes  = len(artifact_loc[artifact_end_trans]) # the detected artifacts, indexes here are the ends of the artifact duration
    total_num          = num_detect_spikes + 1                 # add the first artifact in the artifact_loc, cause diff() will miss the last one
    artifact_end       = np.zeros(total_num,dtype=int)         # prelocate the space for this parameter
    artifact_end[1:]  = artifact_loc[artifact_end_trans+1]
    artifact_end[0]   = artifact_loc[0]                      # add the last artifact index duration head
    """
    plt.figure(100)
    plt.plot(mysignal, hold=True)
    plt.plot(artifact_loc, mysignal[artifact_loc], 'o', color='g', hold=True)    
    plt.plot(artifact_end, mysignal[artifact_end] , 'o', color='r', hold=True)
    threshold_vector  = 5*threshold*np.ones(np.shape(mysignal))      # set a threshold to cross all the stim artifacts
    plt.plot(threshold_vector, color='r', hold= True)
    """
    #plt.plot(real_arti_loc, mysignal[real_arti_loc] , 'o', color='b', hold=True)
    #plt.plot(np.arange(len(mysignal))[head : tail], mysignal[head : tail], color = 'c', hold=True)
    
    total_signal_len = len(mysignal)
    real_arti_loc = np.zeros(len(artifact_end), dtype = int)
    for arti_id in range(len(artifact_end)):

        #arti_id=17
        # based on the roughly detected artifact head, truncate a time window of data with full stimulation artifact in it
        head = artifact_end[arti_id]-half_stim_window   # half_stim_window data points before the artifact_end
        if head<0:   # for those conditions, where the first detected pulse is right after the recording started.
            head = 0 
        tail = artifact_end[arti_id]+half_stim_window   # half_stim_window data points after the artifact_end       
        if tail>total_signal_len:  # for those conditions, where the stim pulse is right before the recording ended.
            tail = total_signal_len
        
        signal_section = mysignal[head : tail]    # this is the section of signal might contain artifact, just based in threshold crossing 
        
        #threshold_vector  = threshold*np.ones(np.shape(signal_section))      # set a threshold to cross all the stim artifacts
       # plt.figure(101)
        #plt.plot(signal_section, hold=True)
       # plt.plot(threshold_vector, color='r', hold=True)
    
        # use the first detected peak as the identification of artifact in this section of signal
        if method == 0:
            # detect the negative going peaks, the amplitude has to be smaller than the threshold  
            peak_loc   = detect_peaks(-signal_section, -threshold)  # mpd: minimum peak distance        
            real_index = peak_loc[0] # always use the first index as the start of the artifact
            # use the lowest peak as the identification of artifact in this section of signal
        elif method == 1:               
            real_index = np.squeeze(np.where(signal_section == min(signal_section)))     # using the lowest point as the real artifact location                    
        real_arti_loc[arti_id]= head + real_index     # the real index in the original time series, because of the diff() used earlier in this function 
      
    return real_arti_loc    
   
"""
plt.figure(50)
plt.subplot(2,1,1)
plt.plot(signal, hold= True)
plt.plot(artifact_end, signal[artifact_end], 'o')
plt.subplot(2,1,2)
plt.plot(signal, hold= True)
plt.plot(real_arti_loc, signal[real_arti_loc], 'o')


plt.figure(40)
plt.plot(signal_section, hold=True)
plt.plot(real_index, signal_section[real_index],'o')
"""
    



def detectArtifact(mysignal, length, rate=0.5):
    """
    INPUT:
    mysignal: a numpy array
    the input signal that the user wants to process.
    
    length: an integer,
    the length for each time window, i.e., int(0.02*fs) means 20 ms is time window.
    This is used to calculate the RMS dynamically, refer to function: dynamicRMS().
    
    rate: a float number,
    the times of the average rms for setting the real threshold to detect possible artifact,
    Default value is 0.5
    i.e., rate=0.5, means 0.5 times of the RMS value. 
    Suggestion, rate is usually setting from 0.5 to 1. 
    
    OUTPUT:
    final_artifact_loc: a numpy array
    Indicate where the artifact locates, the "artifact locations" detected using the original signal and the 
    inverted signal. 
    """ 
    #length = int(0.02*fs)
    rms    = dynamicRMS(mysignal, length)    
    # Define the threhold to detect the artifact
    threshold = -np.mean(rms)*rate   # rate times of the average RMS as the threshold to detect artifact
    artifact_location = artifac_loc(mysignal, threshold) # using threshold to detect artifact
    
    # Invert the input signal and, repeat the above process 
    rms_invert       = dynamicRMS(-mysignal, length)
    threshold_invert = -np.mean(rms_invert)*rate   # one time of the average RMS as the threshold to detect artifact
    #threshold_vector_neg  = -threshold_neg*np.ones(np.shape(mysignal))      # set a threshold to cross all the stim artifacts
    artifact_location_invert = artifac_loc(-mysignal, threshold_invert) # using threshold to detect artifact
    
    # concatenate the two types of locations along the column
    location_matrix =  np.concatenate((np.reshape( artifact_location, (1,len(artifact_location)) ), 
                                       np.reshape(artifact_location_invert, (1, len(artifact_location_invert))) 
                                       ), axis=0)
    # for each artifact, find the smaller index as the final returned artifact location
    final_artifact_loc = np.amin(location_matrix, axis=0)

    return final_artifact_loc
   
"""    
threshold_vector  = threshold*np.ones(np.shape(mysignal))      # set a threshold to cross all the stim artifacts
threshold_vector_invert  = threshold_invert*np.ones(np.shape(mysignal))      # set a threshold to cross all the stim artifacts
plt.figure
plt.plot(mysignal, hold= True)
plt.plot(threshold_vector, hold= True)
plt.plot(artifact_location, mysignal[artifact_location], 'o', color ='r', hold=True)
plt.plot(artifact_location_invert, mysignal[artifact_location_invert], 'o', color = 'b', hold=True)
plt.plot(-threshold_vector_invert, color = 'c', hold=True)

"""





"""
Given the time series, generate its related time vector

time_vector = myTimeVector(signal, fs, time_format)
INPUT:
    signal:      the time series that one wants to analyze 
        fs:      sampling frequency of the time frequency
    time_format: optional input, default option is ms
                 'ms', the returned time is in ms; 
                 's', the return time is in s.       
OUTPUT:
    time_vector: the time vector that is associated with the time series             
"""    
def myTimeVector(*args):
    for count, content in enumerate(args): 
        if count == 0:
            signal = content 
            time_format = 'ms' # default time vector output format in ms
        elif count == 1:
            fs = content
            time_format = 'ms'
        elif count == 2:
            time_format = content 
                
    if time_format == 's':
       time_vector = np.arange(len(signal))/fs  # time vector in s
    elif time_format == 'ms':
        time_vector = np.arange(len(signal))/fs * 1000 # time vector in ms 
    else:
        raise Exception('Invalid time format input! Only accept "ms" or "s".')
        
    return time_vector
    



def EMGextract(EMG, fs=None, artifact=False, delay=0.003, threshold_value=None, TW=None,
               before_artifact=None):
    """
    GIVEN a channel of EMG signal with stim artifact, extract the signal based on stimulation
    signal_matrix  = EMGextract(EMG, fs, threshold, artifact, delay)

    INPUT:
    EMG:(a numpy array, might also work with a list) 
        a channel of EMG signal with stimulation artifact
    fs: (an int number), sampling rate
        threshold_value: custom threshold value used in detecting the stimulation artifact
    artifact: (a boolean), 
        True or Faluse, optional inputs, default is Faluse
        artifact = True, yes, include artifact
        artifact = False, no, don't artifact
    delay: (a float number), 
        in s, optional inputs, default is 3 ms, 
        i.e., stsrting from 3 ms after the artifact to conduct signal extraction. 
        how long after the stim artifact one wants to extract the evoked response 
    TW: (a float number), 
        in s, How long do you want the truncated window to be starting from the "delay"
        i.e, TW = 0.1, 100 ms after the starting of the time window, this is how long
        the user will truncated the signal.
    before_artifact, (a float number), 
        in s. If the user want to truncate the signal including artifact, 
        how long before the artifct does the user wants to include. 
        i.e., before_artifact = 0.01, 10 ms before the artifact.

        
    OUTPUT:
    signal_matrix: extracted signal   
    This is a list, each individual list in this variable is a single extracted evoked response
    """                
    artifact_location = artifac_loc(EMG, threshold_value) # find out the index of the end of the stim artifact
       
    # IF THE USER WANTS TO INCLUDE ARTIFACT OR NOT       
    # the currenct EMG data with stim-to-stim interval of about 400 ms, 400 points per 100 ms, fs = 4k                        
    if artifact:     # yes, include artifact
        before_artifact_end = int(before_artifact*fs)   # 10 ms before the artifact ends
        sig_head = artifact_location - before_artifact_end
    else:           # no, don't need artifact
        window_head = int(delay*fs)         # 3 ms after stimulation artifact starts, trying to not include stim artifact 
        sig_head = artifact_location + window_head

    # extract the recording right after the artifact as signal
    window_tail = int(TW*fs), # 100 ms after the stimulation artifact starts, 
    sig_tail = artifact_location + window_tail
    signal_matrix = [EMG[ sig_head[id]:sig_tail[id] ] for id in range(len(artifact_location)) ] # single line for loop

    return signal_matrix






def SigExtract(signal, artifact_loc, after_artifact_head, TW, 
               artifact=False, before_artifact_head=None):         
    """
    GIVEN a channel of signal with stim artifact, extract the signal based on stimulation
    
    INPUT:
    signal:(a numpy array, might also work with a list) 
        a channel of EMG signal with stimulation artifact
    artifact_loc: (a numpy array)
        this numpy array shows the location of the artifact, should be in index of signal.   
        Refer to function  artifac_loc( )  for more information. 
    artifact: (a boolean), 
        True or Faluse, optional inputs, default is Faluse
        artifact = True, yes, include artifact
        artifact = False, no, don't artifact
    after_artifact_head: (an int number), 
        in data points, optional inputs, default is 3 ms, 
        i.e., stsrting from 3 ms after the artifact to conduct signal extraction. 
        how long after the stim artifact one wants to extract the evoked response 
    TW: (an int number), 
        in data points, How long do you want the truncated window to be starting from the "delay"
        i.e, TW = 0.1, 100 ms after the starting of the time window, this is how long
        the user will truncated the signal.
    before_artifact_head, (an int number), 
        in data points. If the user want to truncate the signal including artifact, 
        how long before the artifct does the user wants to include. 
        i.e., before_artifact_head = 100, 100 data points before the artifact.

        
    OUTPUT:
    signal_matrix: extracted signal   
    This is a list, each individual list in this variable is a single extracted evoked response
    """                
    #artifact_location = artifac_loc(EMG, threshold_value) # find out the index of the end of the stim artifact
       
    # IF THE USER WANTS TO INCLUDE ARTIFACT OR NOT       
    # the currenct EMG data with stim-to-stim interval of about 400 ms, 400 points per 100 ms, fs = 4k                        
    if artifact:     # yes, include artifact
        #before_artifact_head = int(before_artifact*fs)   # 10 ms before the artifact ends
        sig_head = artifact_loc - before_artifact_head
    else:           # no, don't need artifact
        #after_artifact_head = int(delay*fs)         # 3 ms after stimulation artifact starts, trying to not include stim artifact 
        sig_head = artifact_loc + after_artifact_head

    # extract the recording right after the artifact as signal
    sig_tail = artifact_loc + TW
    signal_matrix = [signal[ sig_head[ID]:sig_tail[ID] ] for ID in range(len(artifact_loc)) ] # single line for loop

    return signal_matrix

             
                                                              


"""
generate plot for a given signal
signal, the input signal time series

*argv, this function take various number of input variables, 
       variables are arrange in the order listed below:
       fs, sampling rate
"""
# don't know what is the length of my input,
# the user could just input a signal time series,
# he might also give signal and fs
def plot_sig(*args):
    for count, thing in enumerate(args):
        if count == 0:
           signal = thing
                      
        elif count == 1:
             fs = thing
             time_vector = np.arange(len(signal))/fs * 1000    # time vector in ms                          

    if len(args) == 1:
       plt.figure()
       plt.plot(signal, hold = True, marker = 'o')   # EMG from differential signal of two surface EMG channels
       plt.xlabel('time point')          
       plt.ylabel('amplitude')
    elif len(args) == 2:
        plt.figure()
        plt.plot(time_vector, signal, hold = True, marker = 'o')   # EMG from differential signal of two surface EMG channels
        plt.xlabel('time in ms')  
        plt.ylabel('amplitude')  



"""
conduct FFT for a given signal
Usage:
     (Y, frq) = myfft(y, Fs, length)

INPUT:
     1st input: y,      the time series that you want to conduct the FFT analysis
     2nd input: Fs,     the sampling rate of the signal
     3rd input: length, the given length upon which the user wants to conduct the FFT
                optional variable.

OUTPUT:
    1st output: Y,   Fourier Transform of the original signal, only one side of its frequency
    2nd output: frq, frequency series that is associated with the Fourier Transform
    3rd output: Y1,   Fourier Transform of the original signal, both sides of its frequency

       
"""
def myfft(*args):
    for count, content in enumerate(args):
       
        if count == 0:
           y = content   # the first input is the signal
        elif count == 1:
           Fs = content   # sampling rate of the signal
        elif count == 2:
           length = content  # the length of the FFT that you want to conduct
    
    signal_length = len(y) 
    # only get the the given length of the signal when the length parameter is inputted
    if len(args) == 3:
        if length<=signal_length:
            signal = y[0:length]    # the user speficies a certain length of FFT that he wants to conduct
            signal_length = length  # change the signal length, when the user selected certain signal length
        elif length >signal_length:
            raise ValueError('Length of the FTT computation needs to be smaller than the length of the signal')  
    elif len(args) < 3:
        signal = y
                          
    n = int(signal_length)
    #Y1 = np.fft.fft(signal)/n # fft computing and normalization
    Y1 = np.fft.fft(signal) # fft computing
    Y = Y1[0:round(n/2)] # Fourier Transform of the original signal, only get one side of its frequency range since it is mirror image 

    #Ts = 1.0/Fs; # sampling interval
    #t = np.arange(signal_length)*Ts
    k = np.arange(signal_length)
    T = signal_length/Fs
    frq1 = k/T # two sides frequency range
    frq = frq1[0:round(signal_length/2)] # one side frequency range

    return Y, frq, Y1







def myRMS(mysignal, TW=1):
    """
    calculate the root mean square of a given time series
    INPUT: 
    mysignal, the time series that you want to calculate, a numpy array
    TW,       how many time window does the user wants to break the signal into
    OUTPUT:
    rms,    the rms value of a signal
    """
    n = len(mysignal)   # total length of the input signal
    square = mysignal**2

    # the user wants to preserve the input signal as a whole to calculate its RMS value
    if TW == 1:  
        rms    = np.sqrt(np.sum(square)/n)
    # the user wants to break the input signal into many smaller time windows    
    else:
        rms_values = np.zeros(TW)  # prelocate space to save each RMS values for each small time window
        TW_tail = math.ceil(n/TW)  # floor division
        
        for which_TW in range(0,TW-1): # preserve the last time window and will calculate later
            rms_values[which_TW] = np.sqrt( np.sum(square[which_TW*TW_tail : (which_TW+1)*TW_tail]) /TW_tail )
        
        # calculate the RMS value for the last time window
        rms_values[TW-1] = np.sqrt( np.sum(square[(TW-1)*TW_tail : n])/(n-(TW-1)*TW_tail) )
        rms = np.mean(rms_values)        
 
    return rms



def dynamicRMS(mysignal, length):
    """
    Calculate the RMS value dynamically, for each time window
    INPUT:
    mysignal,  better to be an numpy arrary, 
    the input signal.
    length, in time points   
    the length of each time window that the user wants
    
    OUTPUT: 
    rms, a numpy array.
    """
    num_TW, residual= divmod(len(mysignal), length) # how many time window, and residual data points that this signal has    
    if residual == 0:
        rms = np.zeros(num_TW)
        rms = [ myRMS(mysignal[0+ID*length : (ID+1)*length]) for ID in range(0, num_TW) ]
    elif residual != 0:
        rms = np.zeros(num_TW+1)
        rms = [ myRMS(mysignal[0+ID*length : (ID+1)*length]) for ID in range(0, num_TW+1) ]
    
    rms = np.asarray(rms) # convert the output as a numpy array
    return rms




"""
calculate the rectified power of a signal
INPUT: 
    signal, the time series that you want to calculate, a numpy array
OUTPUT:    
    RecPower, the summation of the rectified signal.
"""
def myRecPower(signal):
    rectify  = abs(signal)
    RecPower = sum(rectify)
    return RecPower





"""
calculate the power of a signal in a given frequency range, based on the summation of the amplitude in frequency range
INPUT: 
    Y,       the Fourier Transform of a signal, complex vectors, refer to the output of function myfft() from above
    frq,     frequency vector coming with Y, refer to the output of function myfft() from above
    f_range, the frequency range that the user wanted 
OUTPUT:    
    f_power, the summation of the power in the given frequency range.    
"""
def myfpower(Y,frq, f_range):

    f_window = np.where(np.logical_and(frq>=f_range[0], frq<=f_range[1]))
    f_power = simps(abs(Y)[f_window]) 
    return f_power





"""
This function is to calculate the evoked amplitude of an evoked response.
A typical evoked response looks like below
NOTE: one can only use this function if it is confirmed that there is some evoked response
           *
          * *          *
         *   *        * *
        *    *       *   *
      *       *     *      *
 *  *          *   *         * * * * 
                *  *     
                *  *   
                * *    
                 *
evoked_amp = myEvokedAmp(evoked_res)
INPUT:
    evoked_res:  an truncated evoked reponse, like the example showing above   
OUTPUT:    
    evoked_amp:  the evoked amplitude calcuated from the evoked response above
"""
def myEvokedAmp(evoked_res):
    evoked_res = evoked_res - np.mean(evoked_res)
    rms_pos = myRMS(evoked_res)  # rms value of the evoked response   
    # rms_neg = myRMS(-evoked_res) # rms value of the evoked response   
    # the positive peaks, there should be two of those
    # return peaks that are higher than rms values threshold, and detected peaks locations has to be larger than 10 points
    peak_loc_pos  =  detect_peaks(evoked_res, mph=rms_pos, mpd=10) 
    # the negative peaks, there should be one negative peak
    peak_loc_neg  =  detect_peaks(evoked_res, mph=rms_pos, mpd=10, valley=1)  # detect valleys
    
    # The evoked response is only described in the shapre as mentioned above
    # if one of the detected peak locations is zero, then there is no evoked response confirmed
    if (peak_loc_neg.size == 0) | (peak_loc_pos.size == 0): 
        if peak_loc_neg.size == 0:
            print('No negative peaks was detected!')
        elif peak_loc_pos.size == 0 :
            print('No positive peaks was detected!')
        evoked_amp = 0  # consider there is no evoked response    
    # only calculate the evoked response when there is a response that is confirmed
    else:   
        # for a typical evoked response, there should be two positive peaks, and one negative peak
        # the evoked response is the two amplitudes averaged with each other
        
        # when there is two positive peaks and one neg peak detected
        if (peak_loc_pos.size == 2) & (peak_loc_neg.size == 1):
            evoked_amp =( abs(evoked_res[peak_loc_pos[0]] - evoked_res[peak_loc_neg]) 
            + abs(evoked_res[peak_loc_pos[1]] - evoked_res[peak_loc_neg]) )/2
        
        # when there is only one positive and one neg peak detected, the pos peak is before neg peak
        elif (peak_loc_pos.size == 1) & (peak_loc_neg.size == 1):
            
            # positive peak is before the negative peak, 
            # consider there is not evoked response 
            if peak_loc_pos < peak_loc_neg:
                # even though, I consider there is no evoked response, I give 10% of the 
                # after stim fluctuation to consider as 'evoked response'
                evoked_amp =0.1*( abs(evoked_res[peak_loc_pos] - evoked_res[peak_loc_neg]) 
                + abs(evoked_res[peak_loc_pos] - evoked_res[-1]))/2
            
            # ps peak is after neg peak, means this is the real evoked response,
            # the first data point in the evoked_res is also a dows slope from a pos peak, 
            # which was not get detected. 
            elif peak_loc_pos > peak_loc_neg: 
                evoked_amp =( abs(evoked_res[peak_loc_pos] - evoked_res[peak_loc_neg]) + abs(evoked_res[peak_loc_neg] - evoked_res[1]))/2
        # in some other cases, where there is just noise, mutiple peaks have been detected.
        else:
            evoked_amp = 0

    return evoked_amp
    
    """
    plt.figure(500)
    plt.plot(evoked_res, hold = True)
    plt.plot(peak_loc_pos, evoked_res[peak_loc_pos],'o' , hold = True)
    plt.plot(peak_loc_neg, evoked_res[peak_loc_neg],'*' , hold = True)
    """





"""
SAVE EMG data into a prenamed pickle files
signal_matrix: row, number of stimulations
               column, time points
visual_label:  the labels for evoked response after visual inspection
               1: yes, there is an evoked EMG response
               0: no, there is no response
"""
def SaveMyEMG(pickle_file, signal_matrix, visual_labels):
    try:
        f = open(pickle_file, 'wb')  # open this data file for writing
        save = {
                'signal_matrix': signal_matrix, # [20 stimulation examples, length in time axis]
                'visual_labels': visual_labels
                }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise





"""
READ EMG data from a presaved pickle files
signal_matrix: row, number of stimulations
               column, time points
visual_label:  the labels for evoked response after visual inspection
               1: yes, there is an evoked EMG response
               0: no, there is no response
"""
def ReadMyPickle(pickle_file):
    try:
        f = open(pickle_file, 'rb')  # open a data file for reading
        data = pickle.load(f)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    signal_matrix = data['signal_matrix']
    visual_labels = data['visual_labels']
  
    return signal_matrix, visual_labels
  



"""
CONDUCT FEATURE ENGINEERING to the EMG data

INPUT:
signal_matrix: row, number of stimulations
               column, time points
Fs: sampling rate

OUTPUT:
f_power:  power in a user selected frequency domain
RMS:      RMS values of the signal in the time domain
p2p:      peak-peak value of the evoked response
RecPower: power of the rectofied signal in the time domain
"""
def EMGFeatureEng(signal_matrix, Fs):
    signal_length = len(signal_matrix[0])  # just take the first signal as an example to judge how long a time series is

    Ts       = 1.0/Fs; # sampling interval
    t        = np.arange(signal_length)*Ts
    t_window = np.squeeze(np.where(t<0.015))      # look at the signal from 0 to 15 ms, when the evoked potential happens
    freq_range = [20, 450]

    f_power  = [] # power in a user selected frequency domain
    RMS      = [] # RMS values of the signal in the time domain
    p2p      = [] # peak-peak value of the evoked response
    RecPower = [] # power of the rectofied signal in the time domain

    for n in range(0, len(signal_matrix)):         
        y = signal_matrix[n] - np.mean(signal_matrix[n])  # shift the signal back to zero baseline
        RMS.append(myRMS(y[t_window]))           # RMS value of the signal
        p2p.append(np.ptp(y[t_window]))          # peak-peak value of a given signal
        RecPower.append(myRecPower(y[t_window])) # power of the rectified signal, the summation of the signal
        (Y, frq, _) = myfft(y, Fs, 256)
        f_power.append(myfpower(Y,frq, freq_range)) # EMG power in a given frequency range. 
    
    # put EMG feature into a dictionary
    EMG_feature = {}
    EMG_feature['f_power'] = f_power
    EMG_feature['RecPower']= RecPower
    EMG_feature['RMS']     = RMS
    EMG_feature['p2p']     = p2p

    return EMG_feature





"""
Normalize the features in the input data
Within each feature, the data will be normalized by minus the mean, and devide by the STD of that feature
INPUT:
    features: the input DataFrame, the field names are the features; 
              the index is the number of observations.
OUTPUT:
    new_features: the output DataFrame, the field names are the features;
                  the index is the number of observations.     
"""
def RescaleFeature(features):
    new_features  = pd.DataFrame(data=features) 

    field_names = features.keys()  # get all the feature names
    for feature_id in range(0, len(field_names)):
        one_feature    = features[field_names[feature_id]]  # get one feature
        normal_feature = (one_feature - np.mean(one_feature))/np.std(one_feature) # normalize or rescale the feature
        new_features[field_names[feature_id]] = normal_feature

    return new_features





"""
Create a butterworth filter to remove some noise
INPUT:
    signal_org: input signal, should a time series
    Fs:         sampling rate
    cut_off:   cut off frequency for the filtering
    filter_typr: what type of filtering we want to do, refer to python function signal.butter() for more info
    vis:         do you want to visualize the designed filter
    
OUTPUT:
    signal_filt: the signal after filtering
"""
def myFilter(*args):
    
    for count, content in enumerate(args):
       
        if count == 0:
           signal_org = content   # the first input is the signal
        elif count == 1:
           Fs = content   # samplinf rate of the signal
        elif count == 2:
           cut_off = content  # the length of the FFT that you want to conduct
        elif count == 3:
           filter_type = content  # the length of the FFT that you want to conduct
        elif count == 4:
           vis = content  # the length of the FFT that you want to conduct
    
    if len(args) != 5: # if the input argument is less 4, then the user don't wnat to visualize the plot
        vis = 0
    
    # a high pass filter to remove low frequency drifting
    nyq = 0.5*Fs
    wn  = cut_off/nyq  # convert it into angular frequency
    b, a = signal.butter(4, wn, filter_type)  
    signal_filt = signal.filtfilt(b, a, signal_org)

    if vis == 1:
        plt.figure()
        w, h = signal.freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(wn, color='green') # cutoff frequency
        plt.show()


    return signal_filt



def MovAvg(signal, MA_win = 2):
    """
    The function takes an input signal, and conduct the moving average of the signal, the moving average window
    is given by the user. At the beginning of the window, when there is not enough data points as the window 
    requires, the function average from the first data point to the current data point; when there is enough
    data point availbe compared to the entire required time window, the function average the time window worth
    of data, and save the data point as the current data point for the processed time series. 
    
    signal, the input signal that user wants to analyze
    MA_win, the time window that the user wants to conduct the moving average,
            default setting is 2.
        
    MA_signal, the processed signal after moving avergae was conducted
    """    
    MA_signal = np.zeros(signal.shape)
    # the default moving average time window is 2
    if MA_win == 2:
        for n in range(0, len(signal)):
            if n< MA_win-1:
                MA_signal[n] = signal[n]/MA_win # when there is no enough data point as the MA_win requires
            else:
                MA_signal[n] = np.sum(signal[n-1:n+1]) /MA_win # averga 2 data points as required

    else: # the user gives a moving average time window
        """
        MA_signal[0] = signal[0] # when there is not enough data points as the MA_win requires 
        for n in range(1, len(signal)):
            if n < MA_win-1:
                MA_signal[n] = np.sum(signal[0:n]) /(n+1) # when there is no enough data point as the MA_win requires
            elif n >= MA_win-1:
                MA_signal[n] = np.sum(signal[n-MA_win : n]) /MA_win # averga data points as required time window
        """
        
        #MA_signal[0] = signal[0] # when there is not enough data points as the MA_win requires 
        for n in range(0, len(signal)):
            if n < MA_win-1:
                MA_signal[n] = np.sum(signal[0:n]) /MA_win # when there is no enough data point as the MA_win requires
            elif n >= MA_win-1:
                MA_signal[n] = np.sum(signal[n-MA_win : n]) /MA_win # averga data points as required time window

    return MA_signal




######## CLASS PlotFigure WITH ALL OF HIS METHODS FOR PLOT DIFFERENT TYPES OF FIGURES #######

class PlotFigure:
    
    '''
    Everytime create an instance of the object for class PlotFigure, 
    python will run it through the first method __int__()
    '''
    def __int__(self, signalx, signaly=None):
        self.signalx = signalx
        self.signaly = signaly
    
    
    def plot_linear_regress(self, performance = False):
        '''
        given data signalx and signaly, 
        create the linear regression of signalx, and plot the regression line on
        top of the raw data points
        
        If you wish to see the print out of the performance of the predict,
        set performance = True;
        Otherwise, it is default value is false.
        '''
        signalx = self.signalx 
        signaly = self.signaly
        
        # make sure both numpy arrays have finite numbers
        finite_ID = np.where(np.isfinite(signalx) & np.isfinite(signaly))

        # reshape the data to meet the format requirement for model fitting,
        # [# samples, # features] 
        plotx = np.reshape(signalx[finite_ID], (len(signalx[finite_ID]),1))
        ploty = np.reshape(signaly[finite_ID], (len(signaly[finite_ID]),1))

        # use linear regress to find the model fit line
        linear_regressor = LinearRegression()
        linear_regressor.fit(plotx, ploty)
        Y_predict = linear_regressor.predict(plotx)

        # plot the original data points and its linear regression line
        plt.figure()
        plt.scatter(plotx, ploty, marker = 'o', hold=True)
        plt.plot(plotx, Y_predict, color = 'red')
        plt.title('Linear regression with raw data')
        
        # plot residual plot
        residual = ploty - Y_predict
        plt.figure()
        plt.scatter(Y_predict, residual, color = 'red',  marker = 'o', hold=True)
        y_zeros = np.zeros(Y_predict.shape) 
        plt.plot(np.sort(np.squeeze(Y_predict)), y_zeros, color = 'black', hold=True)
        plt.xlabel('Fitted value')
        plt.ylabel('Residual')
        plt.title('Versus Fits')
        
        # print model performance as user requires
        if performance:
            rmse = metrics.mean_squared_error(ploty, Y_predict)
            print('Root Mean Squrea Error is ' + str(rmse) + '\n')
            r2 = metrics.r2_score(ploty,Y_predict)
            print('R2 score is: ' + str(r2) + '\n')
    
    
    def plot_polynomial_regress(self, order=2, performance = False):
        '''
        given data signalx and signaly, 
        create the polynomial regression of signalx, and plot the regression line on
        top of the raw data points
        Default order of polynomial fit is 2nd order, user can change it to higher order it he/she wants
        '''
        signalx = self.signalx 
        signaly = self.signaly
        # make sure both numpy arrays have finite numbers
        finite_ID = np.where(np.isfinite(signalx) & np.isfinite(signaly))

        # reshape the data to meet the format requirement for model fitting,
        # [# samples, # features] 
        plotx = np.reshape(signalx[finite_ID], (len(signalx[finite_ID]),1))
        ploty = np.reshape(signaly[finite_ID], (len(signaly[finite_ID]),1))

        # use polynomial regress to find the best fit line
        polynomial_features= PolynomialFeatures(degree=order)  # 2nd order of polynomial fit by default
        x_poly = polynomial_features.fit_transform(plotx)
    
        model = LinearRegression()
        model.fit(x_poly, ploty)
        y_poly_pred = model.predict(x_poly)
    
        plt.figure()
        plt.scatter(plotx, ploty, marker = 'o', hold=True)               
        sorted_plotx = np.sort(plotx, axis=0)
        sorted_ID    = np.argsort(np.squeeze(plotx))      
        sorted_ploty = y_poly_pred[sorted_ID]        
        plt.plot(sorted_plotx, sorted_ploty, linestyle = 'dashed', color ='red')
        plt.title(str(order) + 'nd order polynomial regression with raw data')
       
        # plot residual plot
        residual = ploty - y_poly_pred
        plt.figure()
        plt.scatter(y_poly_pred, residual, color = 'red',  marker = 'o', hold=True)
        y_zeros = np.zeros(y_poly_pred.shape) 
        plt.plot(np.sort(np.squeeze(y_poly_pred)), y_zeros, color = 'black', hold=True)
        plt.xlabel('Fitted value')
        plt.ylabel('Residual')
        plt.title('Versus Fits')
               
        # print model performance as user requires
        if performance:
            rmse = np.sqrt(metrics.mean_squared_error(ploty,y_poly_pred))
            r2 = metrics.r2_score(ploty,y_poly_pred)
            print('Root Mean Squrea Error is: ' + str(rmse) + '\n')
            print('R2 score is: ' + str(r2) + '\n')
               
            
    def yes_no_his(self, first='', second=''):
        '''
        input_yes_no: an input string with two types, i.e. yes and no
        self.signalx: the input data should be a list of strings
        
        first: user wants to define what is the string to be displayed in 
                first column in the figure
        second: user wants to define what is the string to be displayed in 
                the second column in the figure
        '''
        input_yes_no = self.signalx
               
        loc_array = np.zeros(input_yes_no.shape, dtype=float)
        
        # if user never defined what first and second string should be
        # then let the program to select the first and second string to be displayed in the figure
        if not first or not second:        
            first = list(set(input_yes_no))[0]  # what is the first element in the set
            second = list(set(input_yes_no))[1] # what is the 2nd element in the set
        
        for pos, val in enumerate(input_yes_no):
            if val == first:
                loc_array[pos] = float(0)
            elif val == second:
                loc_array[pos] = float(1)
        plt.figure()
        plt.hist(loc_array, align='left')
        plt.xticks([0, 1], [first, second])        
            
            
            
            
            
            
            
            
            