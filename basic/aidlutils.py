# import modules
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.keras.datasets import mnist

'''
print("Module Loaded.")
print("NumPy Version :{}".format(np.__version__))
#print("TensorFlow Version :{}".format(tf.__version__))
print("Matplotlib Version :{}".format(plt.matplotlib.__version__))
'''

# Accuracy
def Accuracy(y:np.ndarray, t:np.ndarray)->np.float32:
    return np.mean(np.equal(np.argmax(y, axis=1).reshape((-1, 1)),t).astype(np.float32))

'''
    결과 출력 함수
    Make_Result_Plot
    Arguments:
        suptitle:
        data:
        label:
        y_max:
'''
def Make_Result_Plot(suptitle:str, data:np.ndarray, label:np.ndarray, y_max:np.ndarray):
    fig_result, ax_result = plt.subplots(2,5,figsize=(18, 7))
    fig_result.suptitle(suptitle)
    for idx in range(10):
        ax_result[idx//5][idx%5].imshow(data[idx].reshape((28,28)),cmap="binary")
        ax_result[idx//5][idx%5].set_title("test_data[{}] (label : {} / y : {})".format(idx, 
                                                                    label[idx], y_max[idx]))


'''
    show image function
    input: 
        x: np.ndarray image data
        y: display image number
'''
def show_img(x, y=16):
    size_img = 28
    plt.figure(figsize=(8,7))
    num_images = y
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img, size_img)
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(x[i], cmap='gray')
    plt.show()
