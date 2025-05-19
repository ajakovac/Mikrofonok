import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks
import sounddevice as sd
import matplotlib.colors as mcolors

class Spectogram:
    def __init__(self, **kwargs):
        self.sample_rate = kwargs.get("sample_rate", 44100) # sampling rate
        self.Noctave     = kwargs.get("octave_resolution", 24) # octave resolution
        self.numin       = kwargs.get("smallest_frequency", 10)  # Hz, the smallest frequency
        self.numax       = kwargs.get("largest_frequency", 20000) # Hz, the largest frequency
        self.tint        = kwargs.get("integral_window", 0.1) #sec
        self.linewidth   = kwargs.get("linewidth", 1) # relative line width in Noctave resolution
        self.timestep    = kwargs.get("time_step", 0.1) # sec, time step for the spectrogram calculation
        self.normalize_spectrum = kwargs.get("normalize_spectrum", True) # normalize the spectrum
        self.noice_reduction_constant = kwargs.get("noice_reduction_constant", 0.1) # constant for the noise reduction

        self.Nnus = int(self.nu_to_index(self.numax)/self.Noctave)*self.Noctave # number of frequency values we consider
        self.frequencies = np.array([ self.index_to_nu(k) for k in  range(self.Nnus)])
        self.Nint = int(self.tint*self.sample_rate)
        self.K = np.empty((self.Nnus,self.Nint))
        self.L = np.empty((self.Nnus,self.Nint))
        self.spectrum_data = None
        self.data = None
        self.duration = None

        # initialize the psectral matrices
        rfactor = 2**(1/self.Noctave)-1
        self.sampling_dt = 1/self.sample_rate            # sampling time step
        for i,oo in enumerate(self.frequencies):
            f = np.array([ np.exp(-(self.linewidth*rfactor*oo*n*self.sampling_dt)**2/2) for n in range(self.Nint) ])
            self.K[i] = np.array([ f[n]*np.sin(2*np.pi*oo*n*self.sampling_dt)*oo for n in range(self.Nint) ])
            self.L[i] = np.array([ f[n]*np.cos(2*np.pi*oo*n*self.sampling_dt)*oo for n in range(self.Nint) ])

    def data_slice(self, t1,t2, data=None):
        if data is None:
            data =self.data
        n1 = int(t1*self.sample_rate)
        n2 = int(t2*self.sample_rate)
        return data[n1:n2].copy()

    def nu_to_index(self, nu) -> float:
        return np.log2(nu/self.numin)*self.Noctave

    def index_to_nu(self, index) -> float:
        return 2**(index/self.Noctave)*self.numin

    def time_to_index(self, rate, *t):
        indices = []
        for ti in t:
            if ti is None:
                indices.append(int(self.duration*rate))
            elif ti>self.duration:
                print(f'time {ti} is out of range: it must be in [0,{self.duration}]')
                indices.append(int(self.duration*rate))
            elif ti< 0:
                print(f'time {ti} is out of range: it must be in [0,{self.duration}]')
                indices.append(0)
            else:
                indices.append(int(ti*rate))
        if len(indices) == 1:
            return indices[0]
        else:
            return tuple(indices)

    def calculate_spectrum(self, n):
        my_spectrum = (self.K@self.data[n:n+self.Nint]/self.Nint)**2 + (self.L@self.data[n:n+self.Nint]/self.Nint)**2
        if self.normalize_spectrum:
            norm = self.noice_reduction_constant + np.linalg.norm(my_spectrum, ord=2)
            my_spectrum = my_spectrum/norm
        return my_spectrum

    def set(self, data):
        self.spectrum_data = None
        self.data = data
        self.duration = len(data)/self.sample_rate  # maximal time of the signal
        self.spectrum_data = []
        self.time_range = np.arange(0,self.duration - self.tint, self.timestep)
        for t in self.time_range:
            n = self.time_to_index(self.sample_rate, t)
            self.spectrum_data.append(self.calculate_spectrum(n))
        self.spectrum_data = np.array(self.spectrum_data)
        return self

    def play(self):
        channel_0_int16 = np.int16(self.data / np.max(np.abs(self.data)) * 32767)
        sd.play(channel_0_int16, samplerate=self.sample_rate)
        sd.wait()

    def plot_data(self, ax = None, **kwargs):
        if self.data is None:
            print('data is not set')
            return None
        if ax is None:
            fig, ax = plt.subplots()
        subt = np.arange(len(self.data))/self.sample_rate
        ax.plot(subt, self.data, **kwargs)
        return ax
    
    def plot_spectrum(self, t, ax = None, **kwargs):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        if ax is None:
            fig, ax = plt.subplots()
        spectrum = self.spectrum_data[self.time_to_index(1/self.timestep,t)]
        ax.plot(self.frequencies, spectrum, **kwargs)
        ax.set_xscale('log')
        return ax

    def find_maxima_in_spectrum_by_index(self, n, nmax=10, threshold=0):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        data = self.spectrum_data[n]
        peak = []
        for index in range(1,len(data)-1):
            if data[index]>data[index-1] and data[index]>data[index+1]:
                omega_list = np.arange(index-1,index+2)
                Vlist = data[index-1:index+2]
                T = np.array([omega_list**2, omega_list, np.ones(omega_list.shape)]).T
                coeff = np.linalg.inv(T)@Vlist
                max_position = -coeff[1]/(2*coeff[0])
                max_value = np.array([max_position**2, max_position, 1]) @ coeff
                if max_value >threshold:
                    peak.append([max_position, self.index_to_nu(max_position), max_value])
        return np.array(sorted(peak, key=lambda x: x[2])[::-1])[:nmax]

    def find_maxima_in_spectrum(self, t, nmax=10, threshold=0):
        return self.find_maxima_in_spectrum_by_index(self.time_to_index(1/self.timestep,t),nmax, threshold)

    def find_maxima_in_spectrogram(self, nmax=10, threshold=0):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        peaks = []
        times = []
        for n in range(len(self.spectrum_data)):
            peaks.append(self.find_maxima_in_spectrum_by_index(n, nmax, threshold))
            times.append(n*self.timestep)
        return times, peaks

    def plot_octaves_2D(self, t, ax = None, **kwargs):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        if ax is None:
            fig, ax = plt.subplots()
        res= self.spectrum_data[self.time_to_index(1/self.timestep,t)].copy()
        res = res.reshape(-1,self.Noctave)
        x = np.arange(res.shape[1]+1)
        y = np.arange(res.shape[0]+1)    
        mesh = ax.pcolormesh(x, y, res, **kwargs)
        return mesh
    
    def show_spectrogram(self, ax = None, **kwargs):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        spectrum = self.spectrum_data
        if ax is None:
            fig, ax = plt.subplots()
        colors = ["white", "blue"]  # White for negative, black for positive
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.1*spectrum.max(), vmax=spectrum.max())
        X, Y = np.meshgrid(self.time_range, self.frequencies, indexing="ij")
        mesh = ax.pcolormesh(X,Y,spectrum, shading='auto', cmap=cmap, norm=norm)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_yscale('log')
        return mesh
