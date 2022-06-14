from scipy.io.wavfile import read, write

def writeFile(filename, x, sampleRate):
    write(filename, sampleRate, x)

def loadFile(filename, sampleRate):
    sr, x = read(filename)
    return x
