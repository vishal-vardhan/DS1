from cmath import sinh, tanh, cosh
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from tensorflow.keras import Sequential, Model


SAMPLE_SIZE = 300
NETA_RANGE = (0.7, 0.9)
DELTA_RANGE = (0.08, 0.1)
N_RANGE = (200, 300)
PITCH_RANGE = (200, 300)
LAMBDA_MAX_RANGE = (1500, 2500)
LAMBDA_RANGE = (500, 3500)
NO_OF_VARIABLES = 5
A = 87.7
mu = 255

def P(neta, delta, N, pitch, x, x0):
    """
    Reflected power function
    """
    sqrt = lambda v: complex(v, 0)**0.5

    g = (1.0/(neta*delta))*((x/x0)-1)
    g = g**2

    theta = neta*delta*N*pitch*sqrt(1-g)/x

    nr = sinh(theta)**2
    dr = (cosh(theta)**2)-g

    return abs(nr/dr)


def reflected_samples(neta:float, delta:float, N:float, pitch:float, x0:float):
    """
    returns : numpy array 
    """
    
    # uniformly sample the function
    # generate sample points
    d = (LAMBDA_RANGE[1] - LAMBDA_RANGE[0])/SAMPLE_SIZE
    x = LAMBDA_RANGE[0]
    
    Y = []
    
    neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*neta
    delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*delta
    N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*N
    pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*pitch
    x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*x0

    for _ in range(SAMPLE_SIZE):
        Y.append(P(neta, delta, N, pitch, x, x0))
        x = x + d
    
    return np.array(Y)

def non_linear_filter(x, law='A'):
    if law == 'A':
        return (1+np.log(A*x))/(1+np.log(A))
    else:
        return np.log(1+mu*x)/np.log(1+mu)
    
def inverse(x, law='A'):
    if law == 'A':
        return np.exp(x*(1+np.log(A))-1)/A
    else:
        return ((1+mu)**x - 1)/mu


# plot data samples
with np.load('data_300_samples200000.npz') as data:
    xtrain = data['Xtrain']
    xtest = data['Xtest']
    ytrain = data['Ytrain']
    ytest = data['Ytest']

d = (LAMBDA_RANGE[1] - LAMBDA_RANGE[0])/SAMPLE_SIZE
x_axis = []
start = LAMBDA_RANGE[0]

for _ in range(SAMPLE_SIZE):
    x_axis.append(start)
    start = start + d

############################################################################
# samples of traning set
# n = len(ytrain)
# r = random.sample(range(1, n), 10)

# for x in r:
#     plt.figure(x)

#     plt.plot(x_axis, ytrain[x])
#     plt.xlabel('wavelength in nm')
#     plt.ylabel('power amplitude')
    
#     neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*xtrain[x][0]
#     delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*xtrain[x][1]
#     N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*xtrain[x][2]
#     pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*xtrain[x][3]
#     x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*xtrain[x][4]

#     name = 'neta = {} delta = {} N = {}, pitch = {}, lB = {}'.format(neta, delta, N, pitch, x0)
#     plt.savefig(name, format='jpg')
#     plt.show()

####################################################################################
# test data plots
# n = len(ytest)
# r = random.sample(range(1, n), 10)

# for x in r:
#     plt.figure(x)

#     plt.plot(x_axis, ytest[x])
    # plt.xlabel('wavelength in nm')
    # plt.ylabel('power amplitude')
    
    # neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*xtest[x][0]
    # delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*xtest[x][1]
    # N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*xtest[x][2]
    # pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*xtest[x][3]
    # x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*xtest[x][4]

    # name = 'neta = {} delta = {} N = {}, pitch = {}, lB = {}'.format(neta, delta, N, pitch, x0)
    # plt.savefig(name, format='jpg')
    # plt.show()

######################################################################################
# # summary of direct model
# model = keras.models.load_model('bwd_without_tandem_64x64x64x32_10epoch_0.0446')
# print(model.summary())

######################################################################################
# # summary of fwd model
# model = keras.models.load_model('fwd_256x256x256x256_50epochs_1e-4_adam_200000')
# print(model.summary())

######################################################################################
# # summary of bwd model
# model = keras.models.load_model('bwd_64x64x64x32_50epochs_4e-4_adam_200000')
# print(model.summary())

######################################################################################
# # summary of tandem model
# model = keras.models.load_model('tandem_fwd_256x256x256x256_50epochs_1e-4_adam_bwd_64x64x64x32_50epochs_4e-4_adam_200000')
# print(model.summary())

######################################################################################
# # direct model predictions
# n = len(ytest)
# r = random.sample(range(1, n), 10)
# model = keras.models.load_model('bwd_without_tandem_64x64x64x32_10epoch_0.0446')
# pred = model.predict(ytest)

# for x in r:
#     plt.figure(x)
    
#     a = xtest[x]
#     plt.plot(x_axis, reflected_samples(a[0], a[1], a[2], a[3], a[4]))
#     a = pred[x]
#     plt.plot(x_axis, reflected_samples(a[0], a[1], a[2], a[3], a[4]))

#     plt.xlabel('wavelength in nm')
#     plt.ylabel('power amplitude')
    
#     neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*xtest[x][0]
#     delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*xtest[x][1]
#     N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*xtest[x][2]
#     pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*xtest[x][3]
#     x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*xtest[x][4]

#     name = 'neta = {} delta = {} N = {}, pitch = {}, lB = {}'.format(neta, delta, N, pitch, x0)
#     plt.savefig(name, format='jpg')
#     plt.show()

######################################################################################
# # different model
# # bwd model
# inputs = Input(shape=ytrain.shape[1:], name='bwd_input_layer')
        
# x = Dense(128, activation='relu', name='bwd_hidden_layer1')(inputs)
# x = BatchNormalization()(x)
# x = Dense(128, activation='relu', name='bwd_hidden_layer2')(x)
# x = BatchNormalization()(x)
# x = Dense(128, activation='relu', name='bwd_hidden_layer3')(x)
# x = BatchNormalization()(x)
# x = Dense(32, activation='relu', name='bwd_hidden_layer4')(x)
# x = BatchNormalization()(x)
# x = Dense(5, name='bwd_output_layer', activation='sigmoid')(x)

# bwd = Model(inputs, x)
# bwd.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
# history = bwd.fit(ytrain, xtrain, epochs=2, batch_size=1024, shuffle=True)

# plt.plot(history.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('epochVsLossbwd128x128x128x32x5', format='jpg')
# plt.show()

# # predictions
# n = len(ytest)
# r = random.sample(range(1, n), 10)
# pred = bwd.predict(ytest)


# for x in r:
#     plt.figure(x)
    
#     a = xtest[x]
#     plt.plot(x_axis, reflected_samples(a[0], a[1], a[2], a[3], a[4]))
#     a = pred[x]
#     plt.plot(x_axis, reflected_samples(a[0], a[1], a[2], a[3], a[4]))

#     plt.xlabel('wavelength in nm')
#     plt.ylabel('power amplitude')
    
#     neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*xtest[x][0]
#     delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*xtest[x][1]
#     N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*xtest[x][2]
#     pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*xtest[x][3]
#     x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*xtest[x][4]

#     name = 'neta = {} delta = {} N = {}, pitch = {}, lB = {}'.format(neta, delta, N, pitch, x0)
#     plt.savefig(name, format='jpg')
#     plt.show()


######################################################################################
# # A-law and mu-law
# nxtrain = xtrain
# nytrain = non_linear_filter(ytrain)
# nxtest = xtest
# nytest = non_linear_filter(ytest)

# inputs = Input(shape=ytrain.shape[1:], name='bwd_input_layer')

# x = Dense(64, activation='relu', name='bwd_hidden_layer1')(inputs)
# x = BatchNormalization()(x)
# x = Dense(64, activation='relu', name='bwd_hidden_layer2')(x)
# x = BatchNormalization()(x)
# x = Dense(64, activation='relu', name='bwd_hidden_layer3')(x)
# x = BatchNormalization()(x)
# x = Dense(32, activation='relu', name='bwd_hidden_layer4')(x)
# x = BatchNormalization()(x)
# x = Dense(5, name='bwd_output_layer', activation='sigmoid')(x)

# bwd = Model(inputs, x, name='inverse model')
# bwd.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
# history = bwd.fit(ytrain, xtrain, epochs=10, batch_size=1024, shuffle=True)

# # plot loss vs epoch
# plt.plot(history.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('epochVsLossbwd64x64x64x32x5', format='jpg')
# plt.show()

# # predictions
# n = len(nytest)
# r = random.sample(range(1, n), 10)
# pred = bwd.predict(nytest)


# for x in r:
#     plt.figure(x)
    
#     a = nxtest[x]
#     plt.plot(x_axis, reflected_samples(a[0], a[1], a[2], a[3], a[4]))
#     a = pred[x]
#     plt.plot(x_axis, reflected_samples(a[0], a[1], a[2], a[3], a[4]))

#     plt.xlabel('wavelength in nm')
#     plt.ylabel('power amplitude')
    
#     neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*nxtest[x][0]
#     delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*nxtest[x][1]
#     N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*nxtest[x][2]
#     pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*nxtest[x][3]
#     x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*nxtest[x][4]

#     name = 'neta = {} delta = {} N = {}, pitch = {}, lB = {}'.format(neta, delta, N, pitch, x0)
#     plt.legend(['actual', 'prediction'], loc='upper left')
#     plt.savefig(name, format='jpg')
#     plt.show()

