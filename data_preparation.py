import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

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
img_size = 128
img_array_normalized = cv2.resize(img_array, (img_size, img_size))
plt.imshow(img_array_normalized)
plt.show()

# function to read all images and labels from one folder append them into a list
def create_data(directory, categories=['NORMAL','PNEUMONIA'], img_size = 128):
    data = []

    for category in CATEGORIES:
        path = os.path.join(directory, category) # path to normal and pneumonia dir
        class_num = CATEGORIES.index(category) # label

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

print('create trainingsdata')
data_train = create_data(traindir)
data_test = create_data(testdir)
print('create testdata')
data_val = create_data(valdir)
x_train, y_train = split_data(data_train)
print('create validation data')
x_test, y_test = split_data(data_test)
x_val, y_val = split_data(data_val)

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