import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import scipy

def resize(npArr):
    return np.array([scipy.misc.imresize(image, (224, 224, 3)) for image in npArr])


def plotAllImages(image, adversarial, predict_label, test_label):

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Original ' + str(predict_label) + ' <> ' + str(test_label))
    plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(adversarial / 255)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    difference = adversarial - image
    sum = np.ndarray.sum(difference)
    plt.title('Difference(' + str(sum) + ')')
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()

def plotImage(image):
    plt.figure()
    plt.imshow(image/255)
    plt.axis('off')
    plt.show()


def saveImageTest(imageArr,index):
    path = "img-test"
    im = Image.fromarray(imageArr)
    im.save(path + os.sep + "ts_im_" + str(index) + ".png")
    
def saveAdvImage(imageNpy, rootPath, footer, isTest = True):
    foldName = "advtest" if isTest else "advtrain"
    imageNpy = (imageNpy * 255 / np.max(imageNpy)).astype('uint8')
    im = Image.fromarray(imageNpy,"RGB")
    im.save(rootPath + os.sep + foldName + os.sep + "adv_im_" + footer + ".png")


def saveImageTrain(imageArr, index):
    path = "img-train"
    im = Image.fromarray(imageArr)
    im.save(path + os.sep + "tr_im_" + str(index) + ".png")

# def saveImageAdversarial(imageArr, index):
#     path = "img-advs"
#     imgg1 = imageArr / 255
#     imgg2 = (imgg1 * 255 / np.max(imgg1)).astype('uint8')
#     im = Image.fromarray(imgg2)
#     im.save(path + os.sep + "adv_im_" + str(index) + ".png")

def pathJoin(directory, file):
    load_dir = os.path.join(os.getcwd(), directory)
    return os.path.join(load_dir, file)

def plotAdversarialDistribution(dist):
    plt.bar(range(len(dist)), list(dist.values()), align='center')
    plt.xticks(range(len(dist)), list(dist.keys()))
    plt.show()


