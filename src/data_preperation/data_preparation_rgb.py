import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from skimage.color import rgb2gray

from src.global_variables import DATA_DIR

# function to read all images and labels from one folder append them into a list
def create_data(directory, categories=['NORMAL','PNEUMONIA'], img_size=128):
    data = []

    for category in categories:
        path = os.path.join(directory, category) # path to normal and pneumonia dir
        class_num = categories.index(category) # label

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            blurred_array = cv2.GaussianBlur(img_array,(5,5),cv2.BORDER_DEFAULT)
            resized_array = cv2.resize(blurred_array, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            data.append([resized_array, class_num])

    return data

# function to split images and labels and store them into seperated lists
def split_data(data):
    x = []
    y = []

    for image, label in data:
        x.append(image)
        y.append(label)
    
    return x, y

# function to view distribution of labels
def label_distribution(dataset):

    label_0 = []
    label_1 = []

    for i in range(len(dataset)):
        if dataset[i] == 0:
            label_0.append(dataset[i])
        if dataset[i] == 1:
            label_1.append(dataset[i])
    
    percentage_normal = (len(label_0)/(len(label_0) + len(label_1)))*100
    percentage_pneumonia = (len(label_1)/(len(label_0) + len(label_1)))*100
    print("percentage of normal xrays = {}".format(percentage_normal))
    print("percentage of pneumonia xrays = {}".format(percentage_pneumonia))

def main():
    datadir = DATA_DIR
    traindir = DATA_DIR / 'train'
    testdir = DATA_DIR / 'test'
    valdir = DATA_DIR / 'val'
    CATEGORIES = ['NORMAL','PNEUMONIA']

    rows_healty = [0,1]
    cols_healthy = [0,1,2]
    rows_pneumenia = [2,3]
    cols_pneumenia = [0,1,2]
    path_pneumonia = os.path.join(traindir, 'Normal')
    path_healthy = os.path.join(traindir, 'PNEUMONIA')
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(8, 8))
    fig.suptitle('Compare healthy and pneumenic xrays')
    plt.tight_layout()

    counter = 0
    for row in (rows_healty):
        for col in (cols_healthy):
            img = os.listdir(path_healthy)[counter]
            img_array = cv2.imread(os.path.join(path_healthy, img))
            axs[row, col].imshow(img_array)
            axs[row, col].set_title('Healthy xray: row - {}, col - {}'.format(row, col))
            counter = counter + 1

    counter = 0
    for row in (rows_pneumenia):
        for col in (cols_pneumenia):
            img = os.listdir(path_pneumonia)[counter]
            img_array = cv2.imread(os.path.join(path_pneumonia, img))
            axs[row, col].imshow(img_array)
            axs[row, col].set_title('Pneumenic xray: row - {}, col - {}'.format(row, col))
            counter = counter + 1

    # reshape image to a standardized and smaller size
    img_size = 128
    img_array_normalized = cv2.resize(img_array, (img_size, img_size))
    plt.imshow(img_array_normalized)
    plt.show()

    # create trainings, test and validation dataset
    print('create trainingsdata')
    data_train = create_data(traindir)
    print('create testdata') 
    data_test = create_data(testdir)
    print('create validation data')
    data_val = create_data(valdir)

    # split images and labels
    x_train, y_train = split_data(data_train)
    plt.imshow(x_train[0])
    plt.show()
    x_test, y_test = split_data(data_test)
    x_val, y_val = split_data(data_val)

    # view distribution of labels
    label_distribution(y_train)

    # distribute trainings set more equal by copying the healty images
    # for i in range(len(y_train)):
    #    if y_train[i] == 0:
    #        x_train.append(x_train[i])
    #        y_train.append(y_train[i])

    # view distribution of labels again
    label_distribution(y_train)

    print('datasets created - storing them in a dict')
    all_data_prepared_rgb = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    'x_val': x_val,
    'y_val': y_val
    }

    with open(str(datadir) + '/all_data_prepared_rgb.pickle', 'wb') as handle:
        print('dumping dict')
        pickle.dump(all_data_prepared_rgb, handle)


if __name__ == "__main__":
    main()


