import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.colors as mcolors

def find_maxima(data, xrange=None, nmax=None, threshold=None, interpolate=True):
    if xrange is None:
        xrange=np.arange(len(data))
    T = np.array([xrange**2, xrange, np.ones(xrange.shape)]).T
    ddata = data[1:]-data[:-1]
    max_indices = np.arange(len(ddata)-1)[np.logical_and(ddata[1:]*ddata[:-1]<=0,ddata[:-1]>0)]+1
    if len(max_indices)==0:
        return [],[]
    max_values = data[max_indices]

    if interpolate:
        Islice = np.array([max_indices-1, max_indices, max_indices+1]).T
        coeff =np.array([np.linalg.inv(T[Ii])@data[Ii] for Ii in Islice])
        max_positions = -coeff[:,1]/(2*coeff[:,0])
        max_values = np.diag(coeff@np.array([max_positions**2, max_positions, np.ones(len(max_positions))]))
    else:
        max_positions = xrange[max_indices]

    if threshold is not None:
        max_positions = max_positions[max_values>threshold]
        max_values = max_values[max_values>threshold]
    ii = np.argsort(-max_values)
    if nmax is not None:
        ii = ii[:nmax]
    return max_positions[ii],max_values[ii]

def data_view(data, view):
    try:
        if view == 'power':
            return np.abs(data)**2
        if view == 'spectrum':
            return np.abs(data)
        elif view == 'phases':
            return np.angle(data)
        elif view == 'real':
            return np.real(data)
        elif view == 'imag':
            return np.imag(data)
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
        return fig, ax

def stft_default_window(x, power=6):
    f = lambda z:1/(1+(2*z)**power)
    if x<-1:
        return 0
    elif x<-0.5:
        return 1-f(x+1)
    elif x<0.5:
        return f(x)
    elif x<1:
        return 1-f(x-1)
    else:
        return 0

class STFT:
    def __init__(self, **kwargs):
        self.sample_rate     = kwargs.get("sample_rate", 44100) # sample rate of the data
        self.t_int           = kwargs.get("integral_window", 0.1) #sec, the width of the window function
        self.step_rate       = kwargs.get("step_rate", 0) #relative step size
        self.window_function = kwargs.get("window_function", stft_default_window) #sec, the sampling window function
        
        self.t_step = self.t_int/(2*(self.step_rate+1))
        self.dt = 1/self.sample_rate
        self.N_T = int(self.t_int*self.sample_rate/2)
        self.N_step = int(self.t_step*self.sample_rate)
        xrange = np.arange(-self.N_T, self.N_T)
        self.W_vector = np.array([ self.window_function(n/self.N_T) for n in xrange])
        self.norm = np.sqrt(np.sum(self.W_vector**2)/(2*self.N_T)/2)
        self.frequencies = np.fft.fftfreq(2*self.N_T, d=self.dt)[:self.N_T+1]

    def range(self, t0, t1):
        return np.arange(t0,t1+self.dt, self.dt)

    def __call__(self, data):
        self.times = []
        self.data = []
        for ncenter in range(self.N_T, len(data)-self.N_T, self.N_step):
            self.times.append(ncenter/self.sample_rate)
            self.data.append(np.fft.ifft(self.W_vector*data[ncenter-self.N_T:ncenter+self.N_T])[:self.N_T+1]/self.norm)
        self.times = np.array(self.times)
        self.data = np.array(self.data)
        return self

    def plot_spectrum(self, t, line='-', ax = None, view='power', **kwargs):
        n_time = int((t-self.times[0])/self.t_step)
        data = data_view(self.data[n_time], view)[:-1]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.frequencies[:-1], data, line, **kwargs)
        ax.set_xscale('log')
        return fig, ax

    def plot(self, ax = None, view='spectrum', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        mask = self.frequencies > 0
        alldata = data_view(self.data, view)[:,mask]
        colors = ["white", "blue"]  # White for negative, black for positive
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        X, Y = np.meshgrid(self.times, self.frequencies[mask], indexing="ij")
        mesh = ax.pcolormesh(X,Y, alldata, shading='gouraud', cmap=cmap)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_yscale('log')
        return fig, ax,mesh

    def power(self):
        return np.sum(np.abs(self.data)**2,axis=1)

    def peaks(self, nmax=10, threshold=0, view='spectrum'):
        data = data_view(self.data, view)
        peaks = []
        for spectrum_data in data:
            peaks.append(find_maxima(spectrum_data, nmax, threshold))
        return peaks
    
    def _synthetize_part(self, n):
        fth = self.data[n]*self.norm
        ftrec = np.concatenate( (fth, np.conj(fth[1:-1][::-1]) ), axis=0)
        return np.real(np.fft.fft(ftrec))
    
    def synthetize(self):
        rec_data=np.array([])
        for data_part in self.data:
            new_data = np.real(np.fft.fft(np.concatenate( (data_part, np.conj(data_part[1:-1][::-1]) ), axis=0)))*self.norm
            if len(rec_data)==0:
                rec_data = new_data
            else:
                rec_data = np.concatenate( ( rec_data, np.zeros(self.N_step) ), axis=0)
                add_data = np.concatenate( ( np.zeros(len(rec_data)-2*self.N_T), new_data ), axis=0)
                rec_data += add_data
        return rec_data/(self.step_rate+1)