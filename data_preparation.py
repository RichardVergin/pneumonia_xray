import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

# function to read all images and labels from one folder append them into a list
def create_data(directory, categories=['NORMAL','PNEUMONIA'], img_size = 256):
    data = []

    for category in categories:
        path = os.path.join(directory, category) # path to normal and pneumonia dir
        class_num = categories.index(category) # label

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            resized_array = cv2.resize(img_array, (img_size, img_size))
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
    traindir = os.path.abspath('C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/train')
    testdir = os.path.abspath('C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/test')
    valdir = os.path.abspath('C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/val')
    CATEGORIES = ['NORMAL','PNEUMONIA']

    # read first image of trainingsfolder to furhter analyze
    for category in CATEGORIES:
        trainpath = os.path.join(traindir, category) # path to normal and pneumonia dir
        for img in os.listdir(trainpath):
            img_array = cv2.imread(os.path.join(trainpath,img))
            plt.imshow(img_array)
            plt.show()
            break
        break

    # reshape image to a standardized and smaller size
    img_size = 256
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
    x_test, y_test = split_data(data_test)
    x_val, y_val = split_data(data_val)

    # view distribution of labels
    label_distribution(y_train)

    print('datasets created - storing them in a dict')
    all_data_prepared = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    'x_val': x_val,
    'y_val': y_val
    }

    with open('C:/MyStuff/Kaggle_Practise/datasets/pneumonia_xray/all_data_prepared.pickle', 'wb') as handle:
        print('dumping dict')
        pickle.dump(all_data_prepared, handle)


if __name__ == "__main__":
    main()


