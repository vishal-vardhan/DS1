from fbg import FBGSegment, FBG
import matplotlib.pyplot as plt

# shift in centre wavelength vs grating length
# l = 8000
# pitch = 0.5355
# dn = 0.0005
# n0 = 1.447
# neta = 0.9
# sampleSize = 300
# lo = 1.546
# hi = 1.554

# x = [x for x in range(1, 11)]
# y = []

# for l in x:
#     fbg = FBG(sampleSize=sampleSize)
#     fbg.push(l = l*1000, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#     X, Y = fbg.getFunctionSamples(lo, hi)
    
#     maxIndex = 0
#     for i in range(len(Y)):
#         if Y[maxIndex] < Y[i]:
#             maxIndex = i
#     y.append(X[maxIndex])

# plt.plot(x, y)
# plt.xlabel("Grating length (mm)")
# plt.ylabel("Shift in centre wavelength (um)")
# plt.savefig('effect of grating length on fbg.jpg', format='jpg')
# plt.show()

##############################################################
# centre wavelength vs change in refractive index
# l = 1000
# pitch = 0.5355
# # dn = 0.0005
# n0 = 1.447
# neta = 0.9
# sampleSize = 300
# lo = 1.546
# hi = 1.554

# x = [0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
# y = []

# for dn in x:
#     fbg = FBG(sampleSize=sampleSize)
#     fbg.push(l = l, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#     X, Y = fbg.getFunctionSamples(lo, hi)
    
#     maxIndex = 0
#     for i in range(len(Y)):
#         if Y[maxIndex] < Y[i]:
#             maxIndex = i
#     y.append(X[maxIndex])

# print(x, y)

# plt.scatter(x, y)
# plt.xlabel("Chane in refractive index")
# plt.ylabel("centre wavelength (um)")
# plt.savefig('effect of refrative index change on centre wavelength.jpg', format='jpg')
# plt.show()

# # bandwidth vs dn
# l = 1000
# pitch = 0.5355
# # dn = 0.0005
# n0 = 1.447
# neta = 0.9
# sampleSize = 300
# lo = 1.546
# hi = 1.554

# x = [0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
# y = []

# for dn in x:
#     fbg = FBG(sampleSize=sampleSize)
#     fbg.push(l = l, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#     X, Y = fbg.getFunctionSamples(lo, hi)
    
#     maxIndex = 0
#     for i in range(len(Y)):
#         if Y[maxIndex] < Y[i]:
#             maxIndex = i

#     leftHalfBW = 0
#     for i in range(maxIndex, -1, -1):
#         if Y[i] < Y[maxIndex]/2:
#             leftHalfBW = X[i]
#             break
#     righthalfBW = 0
#     for i in range(maxIndex, len(Y)):
#         if Y[i] < Y[maxIndex]/2:
#             righthalfBW = X[i]
#             break
    
#     y.append(righthalfBW-leftHalfBW)

# print(x, y)

# plt.plot(x, y)
# plt.xlabel("Chane in refractive index")
# plt.ylabel("bandwidth (um)")
# plt.savefig('effect of refrative index change on bandwidth.jpg', format='jpg')
# plt.show()

# # peak reflectivity vs dn
# l = 1000
# pitch = 0.5355
# # dn = 0.0005
# n0 = 1.447
# neta = 0.9
# sampleSize = 300
# lo = 1.546
# hi = 1.554

# x = [0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
# y = []

# for dn in x:
#     fbg = FBG(sampleSize=sampleSize)
#     fbg.push(l = l, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#     X, Y = fbg.getFunctionSamples(lo, hi)
#     y.append(max(Y))

# print(x, y)

# plt.plot(x, y)
# plt.xlabel("Chane in refractive index")
# plt.ylabel("peak reflectivity")
# plt.savefig('effect of refrative index change on peak reflectivity.jpg', format='jpg')
# plt.show()

# # bandwidth vs dn vs grating length
# # l = 1000
# pitch = 0.5355
# # dn = 0.0005
# n0 = 1.447
# neta = 0.9
# sampleSize = 300
# lo = 1.546
# hi = 1.554

# dnArr = [0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
# lArr = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# for dn in dnArr:
#     y = []
#     for l in lArr:
#         fbg = FBG(sampleSize=sampleSize)
#         fbg.push(l = l*1000, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#         X, Y = fbg.getFunctionSamples(lo, hi)

#         maxIndex = 0
#         for i in range(len(Y)):
#             if Y[maxIndex] < Y[i]:
#                 maxIndex = i

#         leftHalfBW = 0
#         for i in range(maxIndex, -1, -1):
#             if Y[i] < Y[maxIndex]/2:
#                 leftHalfBW = X[i]
#                 break
#         righthalfBW = 0
#         for i in range(maxIndex, len(Y)):
#             if Y[i] < Y[maxIndex]/2:
#                 righthalfBW = X[i]
#                 break
        
#         y.append(righthalfBW-leftHalfBW)
#     plt.plot(lArr, y, label=f'dn={dn}')

# plt.xlabel("Grating length (mm)")
# plt.ylabel("bandwidth (um)")
# plt.legend()
# axes = plt.gca()
# axes.set_xlim([1, 9])
# plt.savefig('effect of grating length on bandwidth for different dn.jpg', format='jpg')
# plt.show()

######################################
# # bandwidth vs dn
# l = 1000
# pitch = 0.5355
# dn = 0.0005
# n0 = 1.447
# # neta = 0.9
# sampleSize = 300
# lo = 1.546
# hi = 1.554

# x = [0.7, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.9]
# y = []

# for neta in x:
#     fbg = FBG(sampleSize=sampleSize)
#     fbg.push(l = l, pitch = pitch, dn = dn, n0 = n0, neta = neta)
#     X, Y = fbg.getFunctionSamples(lo, hi)
    
#     maxIndex = 0
#     for i in range(len(Y)):
#         if Y[maxIndex] < Y[i]:
#             maxIndex = i

#     leftHalfBW = 0
#     for i in range(maxIndex, -1, -1):
#         if Y[i] < Y[maxIndex]/2:
#             leftHalfBW = X[i]
#             break
#     righthalfBW = 0
#     for i in range(maxIndex, len(Y)):
#         if Y[i] < Y[maxIndex]/2:
#             righthalfBW = X[i]
#             break
    
#     y.append(righthalfBW-leftHalfBW)

# plt.plot(x, y)
# plt.xlabel("neta")
# plt.ylabel("bandwidth (um)")
# plt.savefig('effect of neta change on bandwidth.jpg', format='jpg')
# plt.show()