from scipy.io.wavfile import read, write

def writeFile(fileName, x, sampleRate):
    write(filename, sampleRate, x)

def loadFile(filename, sampleRate):
    sr, x = read(filename)
    return x
