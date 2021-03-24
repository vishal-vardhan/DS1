from math import exp, cos, pi, sin

# definitions of different apodization functions
def gaussianFunc(x, data, a=4):
    L = data.l
    return exp(-a*((x-L/2)/L)**2)

def hammingFunc(x, data):
    L = data.l
    return 0.54-0.46*cos(2*pi*x/L)

def barthanFunc(x, data):
    L = data.l
    return 0.62-0.48*abs(x/L-0.5)+0.38*cos(2*pi*(x/L-0.5))

def nuttallFunc(x, data):
    L = data.l
    a0 = 0.3635819
    a1 = 0.48917755
    a2 = 0.1365995
    a3 = 0.0106411
    return a0-a1*cos(2*pi*x/L)+a2*cos(4*pi*x/L)-a3*cos(6*pi*x/L)

def raisedCosineFunc(x, data):
    '''
    alpha : raised cosine parameter
    '''
    L = data.l
    alpha = 0.6
    return alpha*(1+cos(pi*(x-L/2)/L))

def sinc(x, data):
    L = data.l
    pitch = data.pitch
    k = (x-L/2)/pitch
    
    if k == 0:
        return 1/pitch
    
    return sin(k)/k
