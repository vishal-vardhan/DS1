import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from cmath import sinh, tanh, cosh
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import pandas as pd

SAMPLE_SIZE = 300
NETA_RANGE = (0.7, 0.9)
DELTA_RANGE = (0.08, 0.1)
N_RANGE = (200, 300)
PITCH_RANGE = (200, 300)
LAMBDA_MAX_RANGE = (1000, 3000)
LAMBDA_RANGE = (500, 3500)
NO_OF_VARIABLES = 5

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


def test_fwd(fwd, x_test, y_test):
    y_pred = fwd.predict(x_test)
    random_samples = [random.randint(0, len(y_pred)-1) for _ in range(10)]

    d = (LAMBDA_RANGE[1] - LAMBDA_RANGE[0])/SAMPLE_SIZE
    x_axis = []
    start = LAMBDA_RANGE[0]

    for _ in range(SAMPLE_SIZE):
        x_axis.append(start)
        start = start + d

    x_axis = np.array(x_axis)

    for i in random_samples:
        plt.figure(i)
        plt.plot(x_axis, y_test[i])
        plt.plot(x_axis, y_pred[i])
        
        # add data for the plot
        x = x_test
        neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*x[i][0]
        delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*x[i][1]
        N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*x[i][2]
        pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*x[i][3]
        x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*x[i][4]

        # title = 'fwd model : neta = {}, delta = {}, N = {}, pitch = {}, lB = {}'.format(neta, delta, N, pitch, x0)

        # plt.title(title)
        plt.xlabel('nm')
        plt.ylabel('power amplitude')
        plt.legend(['actual', 'prediction'], loc='upper left')
        plt.savefig(str(i))
        plt.show()


def test_bwd(bwd, x_test, y_test):
    results = dict()
    results['index'] = []

    results['act-neta'] = []
    results['act-delta'] = []
    results['act-N'] = []
    results['act-pitch'] = []
    results['act-lambdaB'] = []

    results['pred-neta'] = []
    results['pred-delta'] = []
    results['pred-N'] = []
    results['pred-pitch'] = []
    results['pred-lambdaB'] = []


    x_pred = bwd.predict(y_test)
    random_samples = [random.randint(0, len(x_pred)-1) for _ in range(1000)]

    d = (LAMBDA_RANGE[1] - LAMBDA_RANGE[0])/SAMPLE_SIZE
    x_axis = []
    start = LAMBDA_RANGE[0]

    for _ in range(SAMPLE_SIZE):
        x_axis.append(start)
        start = start + d

    x_axis = np.array(x_axis)

    plt.figure(1)
    for i in random_samples:
        x = x_test[i]
        plt.plot(x_axis, reflected_samples(x[0], x[1], x[2], x[3], x[4]))  # actual
        x = x_pred[i]
        plt.plot(x_axis, reflected_samples(x[0], x[1], x[2], x[3], x[4]))  # prediction

        # add data for the plot
        x = x_pred
        neta = NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*x_test[i][0], NETA_RANGE[0]+(NETA_RANGE[1]-NETA_RANGE[0])*x_pred[i][0]
        delta = DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*x_test[i][1], DELTA_RANGE[0]+(DELTA_RANGE[1]-DELTA_RANGE[0])*x_pred[i][1]
        N = N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*x_test[i][2], N_RANGE[0]+(N_RANGE[1]-N_RANGE[0])*x_pred[i][2]
        pitch = PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*x_test[i][3], PITCH_RANGE[0]+(PITCH_RANGE[1]-PITCH_RANGE[0])*x_pred[i][3]
        x0 = LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*x_test[i][4], LAMBDA_MAX_RANGE[0]+(LAMBDA_MAX_RANGE[1]-LAMBDA_MAX_RANGE[0])*x_pred[i][4]



        # title = 'bwd model : neta = ({}, {}), delta = ({}, {}), N = ({}, {}), pitch = ({}, {}), lB = ({}, {})'.format(neta[0], neta[1], delta[0], delta[1], N[0], N[1], pitch[0], pitch[1], x0[0], x0[1])
        results['index'].append(i)
        
        results['act-neta'].append(neta[0])
        results['pred-neta'].append(neta[1])
        
        results['act-delta'].append(delta[0])
        results['pred-delta'].append(delta[1])
        
        results['act-N'].append(N[0])
        results['pred-N'].append(N[1])
        
        results['act-pitch'].append(pitch[0])
        results['pred-pitch'].append(pitch[1])
        
        results['act-lambdaB'].append(x0[0])
        results['pred-lambdaB'].append(x0[1])
        

        # plt.title(title)
        plt.xlabel('nm')
        plt.ylabel('power amplitude')
        plt.legend(['actual', 'prediction'], loc='upper left')
        plt.savefig('vsplots/'+str(i))
        # plt.show()
        plt.clf()

    df = pd.DataFrame.from_dict(results)
    df.to_pickle("vsplots/data.pkl")


def test_vs_plot(bwd, bwd_without_tandem, x_test, y_test):
    """
    tests the model with and without tandem architecture
    """
    x_pred_direct = bwd_without_tandem.predict(y_test)
    x_pred_tandem = bwd.predict(y_test)
    random_samples = [random.randint(0, len(x_pred_direct)-1) for _ in range(20)]

    d = (LAMBDA_RANGE[1] - LAMBDA_RANGE[0])/SAMPLE_SIZE
    x_axis = []
    start = LAMBDA_RANGE[0]

    for _ in range(SAMPLE_SIZE):
        x_axis.append(start)
        start = start + d

    x_axis = np.array(x_axis)

    for i in random_samples:
        plt.figure(i)
        x = x_test[i]
        plt.plot(x_axis, reflected_samples(x[0], x[1], x[2], x[3], x[4]))
        
        x = x_pred_direct[i]
        plt.plot(x_axis, reflected_samples(x[0], x[1], x[2], x[3], x[4]))
        
        x = x_pred_tandem[i]
        plt.plot(x_axis, reflected_samples(x[0], x[1], x[2], x[3], x[4]))
        
        plt.xlabel('wavelength in nm')
        plt.ylabel('power amplitude')
        plt.legend(['actual', 'direct', 'tandem'], loc='upper left')
        plt.savefig(str(i))
        plt.show()


if __name__ == '__main__':
    bwd = keras.models.load_model('bwd_64x64x64x32_50epochs_4e-4_adam_200000')
    # fwd = keras.models.load_model('fwd_256x256x256x256_50epochs_1e-4_adam_200000')
    # bwd_without_tandem = keras.models.load_model('bwd_without_tandem_64x64x64x32_10epoch_0.0404')

    with np.load('data_300_samples200000.npz') as data:
        x_test = data['Xtest']
        y_test = data['Ytest']

    # # test fwd model
    # test_fwd(fwd, x_test, y_test)

    # # test bwd model
    test_bwd(bwd, x_test, y_test)

    # test bwd without tandem
    # test_bwd(bwd_without_tandem, x_test, y_test)

    # test vs plot
    # test_vs_plot(bwd, bwd_without_tandem, x_test, y_test)