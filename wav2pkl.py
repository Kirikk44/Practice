import matplotlib.pyplot as mp
import scipy.io.wavfile as wav
import scipy.signal as sg
import numpy as np
from PIL import Image
import numpy as np

fs, data = wav.read('adele2001.wav')
data = data[:,0]
# trim the first 125 seconds
#first = data[:int(fs*5)]
data2 = data.ravel()
sp, f, t, im = mp.specgram(data2)
ax = mp.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
mp.savefig('adele.png', bbox_inches="tight", transparent=True, pad_inches=0)


arr = mp.imread('adele.png')
arr = arr[:,0]
print(arr.ravel())
print('--------------------------------------')
print(len(data.ravel()))
print('--------------------------------------')
print(data2)
print('--------------------------------------')





wav.write('adele_new.wav', fs, arr)