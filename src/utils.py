import matplotlib.pyplot as plt

def plotTrainTestLoss(history, savePath=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')

    if savePath:
        plt.savefig(savePath, format='jpg')
    plt.show()

