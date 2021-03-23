import numpy as np
from math import pi
import cmath

class FBGSegment:
    def __init__(self, l, pitch, dn, n0, neta):
        '''
            l : length of the fbg segment
            pitch : pitch of the grating
            dn : difference in refractive index
            n0 : base refractive index
            neta : percentage of power in core
        '''
        self.l = l
        self.pitch = pitch
        self.dn = dn
        self.n0 = n0
        self.neta = neta
        self.peakLambda = 2*n0*pitch  # wavelength at which reflectivity is maximum
        self.next = None

    def getTMatrix(self, x):
        neta = self.neta
        dn = self.dn
        n0 = self.n0
        pitch = self.pitch
        l = self.l

        k = (pi*dn*neta)/x
        beta = (2*pi*n0)/x
        beta0 = pi/pitch
        dBeta = beta-beta0
        j = complex(0, 1)
        s = complex(k*k-dBeta*dBeta, 0)**0.5

        T11 = (cmath.exp(-1*j*beta0*l)*(dBeta*cmath.sinh(s*l)+j*s*cmath.cosh(s*l)))/(j*s)
        T22 = T11.conjugate()
        T12 = (cmath.exp(-1*j*beta0*l)*(k*cmath.sinh(s*l)))/(j*s)
        T21 = T12.conjugate()

        return np.array([[T11, T12], [T21, T22]])


class FBG:
    '''
    @desc : a linked list of FBG segments
    note : everything should be in micro meter (lengths)
    '''
    def __init__(self, sampleSize=300):
        self.head = None
        self.tail = None
        self.nSegments = 0
        self.debug = True
        self.gratingLen = 0
        self.sampleSize = sampleSize

    def push(self, l, dn, n0, neta, pitch):
        if self.head:
            self.tail.next = FBGSegment(l, pitch, dn, n0, neta)
            self.tail = self.tail.next
        else:
            self.head = FBGSegment(l, pitch, dn, n0, neta)
            self.tail = self.head

        self.nSegments += 1
        self.gratingLen += l

    def pop(self):
        self.gratingLen -= self.tail.l
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            temp = self.head.next
            self.head.next = None
            self.head = temp

        self.nSegments -= 1

    def __getCombinedTMatrix(self, x):
        '''
        returns product of T matrices of the connected fbg's
        '''
        ptr = self.head
        T = np.eye(2)
        while ptr:
            T = np.matmul(T, ptr.getTMatrix(x))
            ptr = ptr.next
        
        return T

    def R(self, x):
        '''
        reflectivity function samples
        '''
        T = self.__getCombinedTMatrix(x)
        out = np.matmul(T, np.array([[0], [1]]))
        a0 = out[0][0]
        b0 = out[1][0]
        r = abs(a0/b0)**2
        return r
    
    def getFunctionSamples(self, lo, hi):
        x = [0 for _ in range(self.sampleSize)]
        y = [0 for _ in range(self.sampleSize)]
        d = (hi-lo)/self.sampleSize

        for i in range(self.sampleSize):
            x[i] = lo
            y[i] = self.R(x[i])
            lo = lo+d

        return x, y
    
    def setApodization(self, apodFunc):
        ptr = self.head
        x = 0
        while ptr:
            ptr.dn = apodFunc(x+ptr.l/2, ptr)*ptr.dn
            x += ptr.l
            ptr = ptr.next


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from apodization import gaussianFunc, raisedCosineFunc, hammingFunc, barthanFunc, nuttallFunc, sinc

#     l = 8000
#     pitch = 0.5355
#     dn = 0.0005
#     n0 = 1.447
#     neta = 0.9
#     sampleSize = 300
#     lo = 1.546
#     hi = 1.554

#     fbg = FBG(sampleSize=sampleSize)
#     fbg.push(l = l, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#     x, y = fbg.getFunctionSamples(lo, hi)
#     plt.plot(x, y, label=f'L={l/1000}mm')
#     plt.xlabel("wavelength (um)")
#     plt.ylabel("reflectivity")
#     plt.legend()
#     axes = plt.gca()
#     axes.set_ylim([0, 1])
#     plt.savefig(f'L={l/1000}mm.jpg', format='jpg')
#     plt.show()