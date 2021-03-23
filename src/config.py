class Item:
    def __init__(self, minv, maxv):
        self.min = minv
        self.max = maxv

lRange = Item(1000, 2000)  # increasing l would introduce oscillations
dnRange = Item(0.0004, 0.001)
n0Range = Item(1.4415, 1.4415)
netaRange = Item(0.75, 0.95)
pitchRange = Item(0.535, 0.5365)
xRange = Item(1.54, 1.55)
nSamples = 300
nVariables = 5