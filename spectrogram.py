import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks
import sounddevice as sd
import matplotlib.colors as mcolors

def find_maxima(data, nmax=10, threshold=0):
    if data is None:
        return None
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
                peak.append([max_position, max_value])
    return np.array(sorted(peak, key=lambda x: x[1])[::-1])[:nmax]

def data_view(data, view):
    try:
        if view == 'spectrum':
            return data[2]
        elif view == 'phases':
            return data[3]
        elif view == 'real':
            return data[0]
        elif view == 'imag':
            return data[1]
        else:
            print(f'view {view} is not supported')
            return None
    except Exception as e:
        print(f'Error in view function: {e}')
        return None

class SoundData:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sampling_dt = 1/sample_rate
    
    def set(self, data):
        self.data = data
        self.duration = len(data)/self.sample_rate  # maximal time of the signal
        return self

    def time_to_index(self, ti):
        if ti is None:
            return int(self.duration*self.sample_rate)
        elif ti>self.duration:
            print(f'time {ti} is out of range: it must be in [0,{self.duration}]')
            return int(self.duration*self.sample_rate)
        elif ti< 0:
            print(f'time {ti} is out of range: it must be in [0,{self.duration}]')
            return 0
        else:
            return int(ti*self.sample_rate)

    def slice(self, t1,t2):
        n1 = self.time_to_index(t1)
        n2 = self.time_to_index(t2)
        return self.data[n1:n2]

    def play(self, t1=0, t2=None):
        channel_0_int16 = np.int16(self.slice(t1,t2) / np.max(np.abs(self.slice(t1,t2))) * 32767)
        sd.play(channel_0_int16, samplerate=self.sample_rate)
        sd.wait()

    def plot(self, ax = None, t1=0, t2=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        data = self.slice(t1, t2)
        subt = np.arange(len(data))/self.sample_rate
        ax.plot(subt, data, **kwargs)
        return ax

class SpectrumGenerator:
    def __init__(self, sample_rate=44100, **kwargs):
        self.Noctave     = kwargs.get("octave_resolution", 24) # octave resolution
        self.numin       = kwargs.get("smallest_frequency", 20)  # Hz, the smallest frequency
        numax            = kwargs.get("largest_frequency", 20000) # Hz, the potential largest frequency
        self.tint        = kwargs.get("integral_window", 0.1) #sec
        self.linewidth   = kwargs.get("linewidth", 1) # relative line width in Noctave resolution
        self.noice_reduction_constant = kwargs.get("noice_reduction_constant", 0.1) # constant for the noise reduction

        self.Nnus = int(self.nu_to_index(numax)/self.Noctave)*self.Noctave # number of frequency values we consider
        self.frequencies = np.array([ self.index_to_nu(k) for k in  range(self.Nnus)])
        self.reset_matrices(sample_rate)
    
    def reset_matrices(self, sample_rate):
        decay_constant = 3
        sigma0 = decay_constant/self.tint
        self.sample_rate = sample_rate
        sampling_dt = 1/sample_rate            # sampling time step
        self.Nint = int(self.tint*sample_rate)
        self.K = np.empty((self.Nnus,2*self.Nint+1), dtype=complex)
        self.data = None
        # initialize the spectral matrices
        rfactor = 2**(1/self.Noctave)-1
        for i,oo in enumerate(self.frequencies):
            f = np.array([ np.exp(-(self.linewidth*(sigma0 + rfactor*oo)*n*sampling_dt)**2/2) for n in range(-self.Nint, self.Nint+1) ])
            self.K[i] = np.array([ f[n]*np.exp(1j*2*np.pi*oo*n*sampling_dt)*oo for n in range(-self.Nint, self.Nint+1) ])

    def nu_to_index(self, nu) -> float:
        return np.log2(nu/self.numin)*self.Noctave

    def index_to_nu(self, index) -> float:
        return 2**(index/self.Noctave)*self.numin

    def __call__(self, sound_data:SoundData, t):
        if sound_data.sample_rate != self.sample_rate:
            self.sample_rate = sound_data.sample_rate
            print(f'sample rate changed to {self.sample_rate} Hz')
            self.reset_matrices(self.sample_rate)
        if t > sound_data.duration - self.tint:
            print(f'time {t} is out of range: it must be in [0,{sound_data.duration - self.tint}]')
            return None
        n = sound_data.time_to_index(t)
        FT = self.K@sound_data.data[n-self.Nint:n+self.Nint+1]
        self.data = np.array( [FT.real, FT.imag, np.abs(FT), np.angle(FT)] )
        return self
    
    def power(self):
        data = data_view(self.data, "spectrum")
        if data is None:
            return None
        

    def plot(self, ax = None, view='spectrum', **kwargs):
        data = data_view(self.data, view)
        if data is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.frequencies, data, **kwargs)
        ax.set_xscale('log')
        return ax

    def plot_octaves_2D(self, ax = None, view='spectrum', **kwargs):
        data = data_view(self.data, view)
        if data is None:
            return None
        if ax is None:
            fig, ax = plt.subplots()
        res= data.copy().astype(np.float64)
        res = res.reshape(-1,self.Noctave)
        x = np.arange(res.shape[1]+1)
        y = np.arange(res.shape[0]+1)    
        mesh = ax.pcolormesh(x, y, res, **kwargs)
        return mesh

class Spectrogram:
    def __init__(self, spectrum_class:SpectrumGenerator, time_step=0.1):
        self.timestep = time_step  # sec, time step for the spectrogram
        self.spectrum_class = spectrum_class
        self.data = None

    def __call__(self, sound_data:SoundData):
        self.time_range = np.arange(0,sound_data.duration - self.spectrum_class.tint, self.timestep)
        self.data = []
        for t in self.time_range:
            self.data.append(self.spectrum_class(sound_data=sound_data, t=t).data)
        self.data = np.array(self.data)

    def peaks(self, nmax=10, threshold=0, view='spectrum'):
        data = data_view(self.data, view)
        if data is None:
            return None, None
        peaks = []
        times = []
        for n, spectrum_data in enumerate(data):
            peaks.append(find_maxima(spectrum_data, nmax, threshold))
            times.append(n*self.timestep)
        return times, peaks

    def plot(self, ax = None, view='spectrum', **kwargs):
        data = data_view(self.data, view)
        if data is None:
            return None
        data = self.data[view]
        if ax is None:
            fig, ax = plt.subplots()
        colors = ["white", "blue"]  # White for negative, black for positive
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.1*data.max(), vmax=data.max())
        X, Y = np.meshgrid(self.time_range, self.spectrum_class.frequencies, indexing="ij")
        mesh = ax.pcolormesh(X,Y,data, shading='auto', cmap=cmap, norm=norm)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_yscale('log')
        return mesh

class Synthesizer:
    def __init__(self, spectrogram: Spectrogram, envelope_basefunction= lambda x: 1-x**6/(1+x**6)):
        self.spectrogram = spectrogram
        self.spectrum_class = spectrogram.spectrum_class
        self.sound_data = spectrogram.sound_data

        self.timestep = spectrogram.timestep
        self.sampling_dt = self.sound_data.sampling_dt
        self.frequencies = self.spectrum_class.frequencies
        self.ndata = int(self.timestep/ self.sampling_dt)
        x = np.linspace(-1,1,self.ndata+1)
        fx = envelope_basefunction(x)
        self.envelope = np.concatenate(( 1-fx[int(self.ndata/2):-1], fx, 1-fx[1:int(self.ndata/2)]))
        trange = np.arange(0,2*self.timestep, self.sampling_dt)
        self.wavesamples = np.array([ self.envelope*np.exp(-2*1j*np.pi*f*trange) for f in self.frequencies])

    def __call__(self):
        spectrum = self.spectrogram.data['spectrum']
        duration = spectrum.shape[0]*self.timestep   # seconds
        starttimes = np.arange(0, duration, self.timestep)
        phases = np.array( [ [np.exp(-2*1j*np.pi*nu*ti) for nu in self.frequencies] for ti in starttimes] )
        generator = (spectrum*phases)@self.wavesamples

        #from the generator we generate the data
        data = generator[0]

        for gen in generator[1:]:
            data = np.concatenate((data, np.zeros(shape=(self.ndata,)))) + np.concatenate(( np.zeros(data.shape[0]-self.ndata), gen))

        return data.real

