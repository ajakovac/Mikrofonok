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

        self.Nnus = int(np.log2(self.numax/self.numin))*self.Noctave # number of frequency values we consider
        self.frequencies = np.array([ 2**(k/self.Noctave)*self.numin for k in  range(self.Nnus)])
        self.Nint = int(self.tint*self.sample_rate)
        self.K = np.empty((self.Nnus,self.Nint))
        self.L = np.empty((self.Nnus,self.Nint))
        self.spectrum_data = None
        self.data = None
        self.maxtime = None

        # initialize the psectral matrices
        rfactor = 2**(1/self.Noctave)-1
        self.sampling_dt = 1/self.sample_rate            # sampling time step
        for i,oo in enumerate(self.frequencies):
            f = np.array([ np.exp(-(self.linewidth*rfactor*oo*n*self.sampling_dt)**2/2) for n in range(self.Nint) ])
            self.K[i] = np.array([ f[n]*np.sin(2*np.pi*oo*n*self.sampling_dt)*oo for n in range(self.Nint) ])
            self.L[i] = np.array([ f[n]*np.cos(2*np.pi*oo*n*self.sampling_dt)*oo for n in range(self.Nint) ])

    def set(self, data):
        self.spectrum_data = None
        self.data = data
        self.maxtime = len(data)/self.sample_rate  # maximal time of the signal
        return self

    def calculate_spectrum(self, t=0):
        if self.data is None:
            print('data is not set')
            return None
        if t<0 or t+self.tint > self.maxtime:
            print(f'time {t} is out of range: it must be in [0,{self.maxtime}]')
            return None
        n = int(t*self.sample_rate)
        my_spectrum = (self.K@self.data[n:n+self.Nint]/self.Nint)**2 + (self.L@self.data[n:n+self.Nint]/self.Nint)**2
        if self.normalize_spectrum:
            norm = self.noice_reduction_constant + np.linalg.norm(my_spectrum, ord=2)
            my_spectrum = my_spectrum/norm
        return my_spectrum

    def __call__(self, t=0, t_end=None):
        if self.data is None:
            print('data is not set')
            return None
        if t_end is None:
            t_end = self.maxtime - self.tint
        self.spectrum_data = []
        self.time_range = np.arange(t, t_end, self.timestep)
        for t in self.time_range:
            self.spectrum_data.append(self.calculate_spectrum(t))
        self.spectrum_data = np.array(self.spectrum_data)
        return self.time_range, self.spectrum_data

    def convert_range(self, t1, t2):
        n1 = int(t1*self.sample_rate)
        if t2 is not None:
            n2 = int(t2*self.sample_rate)
        else:
            n2 = len(self.data)
        return n1,n2

    def play(self, t1 = 0, t2=None):
        n1,n2 = self.convert_range(t1,t2)
        channel_0_int16 = np.int16(self.data[n1:n2] / np.max(np.abs(self.data)) * 32767)
        sd.play(channel_0_int16, samplerate=self.sample_rate)
        sd.wait()

    def plot_data(self, t1=0, t2=None, ax = None, **kwargs):
        if self.data is None:
            print('data is not set')
            return None
        if ax is None:
            fig, ax = plt.subplots()
        n1,n2 = self.convert_range(t1,t2)
        subt = np.arange(len(self.data[n1:n2]))/self.sample_rate + t1
        ax.plot(subt, self.data[n1:n2], **kwargs)
        return ax
    
    def plot_spectrum(self, t, ax = None, **kwargs):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        if t<0 or t>self.maxtime:
            print(f'time {t} is out of range: it must be in [0,{self.maxtime}]')
            return None
        if ax is None:
            fig, ax = plt.subplots()
        spectrum = self.spectrum_data[int(t/self.timestep)]
        ax.plot(self.frequencies, spectrum, **kwargs)
        ax.set_xscale('log')
        return ax

    def max_freq(self, t, map = (lambda x: np.log(x), lambda x:np.exp(x))):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        if t<0 or t>self.maxtime:
            print(f'time {t} is out of range: it must be in [0,{self.maxtime}]')
            return None
        spectrum = self.spectrum_data[int(t/self.timestep)]
        index = np.argmax(spectrum)
        if index == 0 or index ==len(spectrum)-1:
            return index, spectrum[index]
        omega_list = map[0](self.frequencies[index-1:index+2])
        Vlist = spectrum[index-1:index+2]
        T = np.array([omega_list**2, omega_list, np.ones(omega_list.shape)]).T
        coeff = np.linalg.inv(T)@Vlist
        return map[1](-coeff[1]/(2*coeff[0]))

    def plot_octaves_2D(self, t, ax = None, **kwargs):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        if t<0 or t>self.maxtime:
            print(f'time {t} is out of range: it must be in [0,{self.maxtime}]')
            return None
        if ax is None:
            fig, ax = plt.subplots()
        res= self.spectrum_data[int(t/self.timestep)].copy()
        res = res.reshape(-1,self.Noctave)
        x = np.arange(res.shape[1]+1)
        y = np.arange(res.shape[0]+1)    
        mesh = ax.pcolormesh(x, y, res, **kwargs)
        return mesh
    
    def show_spectrogram(self, t1=0, t2=None, ax = None, **kwargs):
        if self.spectrum_data is None:
            print('spectrum is not calculated')
            return None
        if t2 is None:
            t2 = self.maxtime
        if ax is None:
            fig, ax = plt.subplots()
        spectrum = self.spectrum_data[int(t1/self.timestep):int(t2/self.timestep)]
        colors = ["white", "blue"]  # White for negative, black for positive
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.1*spectrum.max(), vmax=spectrum.max())
        X, Y = np.meshgrid(self.time_range, self.frequencies, indexing="ij")
        mesh = ax.pcolormesh(X,Y,spectrum, shading='auto', cmap=cmap, norm=norm)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_yscale('log')
        return mesh
