import scipy.io.wavfile as wav
import numpy as np
import librosa
import pickle


class conf:
    sampling_rate = 44100
    duration = 5  # sec
    hop_length = 347 * duration  # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration


def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples:  # long enough
        if trim_long_data:
            y = y[0:0 + conf.samples]
    else:  # pad blank
        padding = conf.samples - len(y)  # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)
    return y


def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    # if debug_display:
    #     IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
    #     show_melspectrogram(conf, mels)
    return mels


def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def save_as_pkl_binary(obj, filename):
    """Save object as pickle binary file.
    Thanks to https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)


def get_default_conf():
    return conf


def divideAudio(audioname, partSize, conf):
    fs, data = wav.read(audioname)
    data = data[:, 0]
    i = partSize
    j = 0
    X = []
    while (int(i * fs) <= len(data)):
        part = data[int(fs * j):int(fs * i)]
        wav.write(f'shorts/{audioname}{i / partSize}.wav', fs, part)
        x = read_as_melspectrogram(conf, f'shorts/{audioname}{i / partSize}.wav', trim_long_data=False)
        x_color = mono_to_color(x)
        X.append(x_color)
        #print(f'{i} is done')

        i = i + partSize
        j = j + partSize
    if int(j * fs) != len(data):
        i = len(data)
        part = data[int(fs * j):i]
        wav.write(f'shorts/{audioname}{int(i / (partSize * fs)) + 1}.wav', fs, part)
    return X

_conf = get_default_conf()
X = divideAudio('longPlay3.wav', 5, _conf)
save_as_pkl_binary(X, 'longPlay3.pkl')
