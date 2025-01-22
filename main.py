import os
import numpy as np
from skimage.feature import hog
import sys
import matplotlib.pyplot as plt


from A.dummy_classifier import dummy_classifier_model
from A.random_forest import random_forest_model
from A.svm_task1 import svm_model_task1
from A.cnn_task1 import cnn

from B.svm_task2 import svm_model_task2
from B.CNN_task2 import cnn_task2

def reading_datafile_task1():
    # Navigate to the dataset file
    npz_file_path = os.path.join('Datasets', 'breastmnist.npz')    # Navigate to the dataset file
    data = np.load(npz_file_path)          #  Load the breastmnist.npz file using numpy

    print("Stored arrays:", data.files)   # Print the names of arrays stored in the .npz file

    train_dataset = data['train_images']
    print('Training dataset', train_dataset.shape)

    val_dataset = data['val_images']
    print('Validation dataset', val_dataset.shape)

    test_dataset = data['test_images']
    print('Testing dataset', test_dataset.shape)

    y_train_labels = data['train_labels']
    print('Training label', y_train_labels.shape)


    y_val_labels = data['val_labels']
    print('Validation label', y_val_labels.shape)


    y_test_labels = data['test_labels']
    print('Testing label', y_test_labels.shape)

    return train_dataset, val_dataset, test_dataset, y_train_labels, y_val_labels, y_test_labels

def reading_datafile_task2():
    
    npz_file_path_t2 = os.path.join('Datasets', 'bloodmnist.npz') # Navigate to the dataset file
 
    data_t2 = np.load(npz_file_path_t2)     # Load the bloodmnist.npz file using numpy

    print("Stored arrays:", data_t2.files)   # Print the names of arrays stored in the .npz file

    train_dataset_task2 = data_t2['train_images']
    print('Training dataset', train_dataset_task2.shape)

    val_dataset_task2 = data_t2['val_images']
    print('Validation dataset', val_dataset_task2.shape)

    test_dataset_task2 = data_t2['test_images']
    print('Testing dataset', test_dataset_task2.shape)

    y_train_labels_task2 = data_t2['train_labels']
    print('Training label', y_train_labels_task2.shape)


    y_val_labels_task2 = data_t2['val_labels']
    print('Validation label', y_val_labels_task2.shape)


    y_test_labels_task2 = data_t2['test_labels']
    print('Testing label', y_test_labels_task2.shape)

    return train_dataset_task2, val_dataset_task2, test_dataset_task2, y_train_labels_task2, y_val_labels_task2, y_test_labels_task2




def flatten_dataset(train_dataset, val_dataset, test_dataset, y_train_labels, y_val_labels, y_test_labels):
    # Training
    x_train = train_dataset.reshape(train_dataset.shape[0], -1)
    y_train_labels=y_train_labels.flatten()

    # Preprocessing val
    x_val = val_dataset.reshape(val_dataset.shape[0], -1)
    y_val_labels=y_val_labels.flatten()

    # Preprocessing test
    x_test = test_dataset.reshape(test_dataset.shape[0], -1)
    y_test_labels=y_test_labels.flatten()

    return x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels


def normalise_data(x_train, x_val, x_test):   # Normalise the data
    #Normalise train
    x_train = np.array(x_train) 
    x_train = x_train / 255.0   # Normalise the pixel values to range [0, 1] by dividing by 255
        
    #Normalise validation
    x_val = np.array(x_val) 
    x_val = x_val / 255.0  # Normalise the pixel values to range [0, 1] by dividing by 255

    #Normalise test
    x_test = np.array(x_test)  #
    x_test = x_test / 255.0    # Normalise the pixel values to range [0, 1] by dividing by 255

    return x_train, x_val, x_test

def hog_data_task1(train_dataset, val_dataset, test_dataset):    # Perform HOG feature extraction
    sample_hog_features= hog(
        train_dataset[0], 
        block_norm='L2',
        orientations=9, 
        pixels_per_cell=(7, 7), 
        cells_per_block=(2, 2), 
        visualize=False, 
        feature_vector=True
    )

    # Number of features per image
    num_features = len(sample_hog_features)


    # Train
    x_features_train = np.empty((len(x_train),num_features))  # Assuming each image has 324 HOG features

    # Loop through each image in the dataset
    for x in range(len(x_train)):
        # Compute the HOG features for the current image
        hog_features, hog_image = hog(
            train_dataset[x],  # Process the current image in the loop
            block_norm='L2',
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True
        )
        # Store the HOG features for the current image in the x_features array
        x_features_train[x] = hog_features  # Assign the feature vector for each image

    #Valid
    x_features_val = np.empty((len(val_dataset), num_features))  # Assuming each image has 324 HOG features

    # Loop through each image in the dataset
    for x in range(len(val_dataset)):
        # Compute the HOG features for the current image
        hog_features, hog_image = hog(
            val_dataset[x],  # Process the current image in the loop
            block_norm='L2',
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True
        )
        # Store the HOG features for the current image in the x_features array
        x_features_val[x] = hog_features  # Assign the feature vector for each image

    #Test
    x_features_test = np.empty((len(test_dataset), num_features))  # Assuming each image has 324 HOG features

    # Loop through each image in the dataset
    for x in range(len(test_dataset)):
        # Compute the HOG features for the current image
        hog_features, hog_image = hog(
            test_dataset[x],  # Process the current image in the loop
            block_norm='L2',
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True
        )
        # Store the HOG features for the current image in the x_features array
        x_features_test[x] = hog_features  # Assign the feature vector for each image

    return x_features_train, x_features_val,x_features_test


def hog_data_task2(train_dataset, val_dataset, test_dataset):    # Perform HOG feature extraction
    sample_hog_features= hog(
        train_dataset[0], 
        block_norm='L2',
        orientations=9, 
        pixels_per_cell=(7, 7), 
        cells_per_block=(2, 2), 
        visualize=False, 
        feature_vector=True,
        channel_axis=-1
    )

    # Number of features per image
    num_features = len(sample_hog_features)


    # Train
    x_features_train = np.empty((len(train_dataset),num_features))  # Assuming each image has 324 HOG features

    # Loop through each image in the dataset
    for x in range(len(train_dataset)):
        # Compute the HOG features for the current image
        hog_features, hog_image = hog(
            train_dataset[x],  # Process the current image in the loop
            block_norm='L2',
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True,
            channel_axis=-1
        )
        # Store the HOG features for the current image in the x_features array
        x_features_train[x] = hog_features  # Assign the feature vector for each image

    #Valid
    x_features_val = np.empty((len(val_dataset), num_features))  # Assuming each image has 324 HOG features

    # Loop through each image in the dataset
    for x in range(len(val_dataset)):
        # Compute the HOG features for the current image
        hog_features, hog_image = hog(
            val_dataset[x],  # Process the current image in the loop
            block_norm='L2',
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True,
            channel_axis=-1
        )
        # Store the HOG features for the current image in the x_features array
        x_features_val[x] = hog_features  # Assign the feature vector for each image

    #Test
    x_features_test = np.empty((len(test_dataset), num_features))  # Assuming each image has 324 HOG features

    # Loop through each image in the dataset
    for x in range(len(test_dataset)):
        # Compute the HOG features for the current image
        hog_features, hog_image = hog(
            test_dataset[x],  # Process the current image in the loop
            block_norm='L2',
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True,
            channel_axis=-1
        )
        # Store the HOG features for the current image in the x_features array
        x_features_test[x] = hog_features  # Assign the feature vector for each image

    return x_features_train, x_features_val,x_features_test





if __name__ == '__main__':

    ######### Task 1 ##############################
    print("********************************************* Task 1***********************************************")
    train_dataset, val_dataset, test_dataset, y_train_labels, y_val_labels, y_test_labels  = reading_datafile_task1()    #read from .npz datafile
    x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels = flatten_dataset(train_dataset, val_dataset, test_dataset, y_train_labels, y_val_labels, y_test_labels)
    x_train_data, x_val_data, x_test_data = normalise_data(x_train, x_val, x_test)
    x_features_train, x_features_val,x_features_test = hog_data_task1(train_dataset, val_dataset, test_dataset)

    #class distribution
    malignant_count = 0
    benign_count = 0
    # Iterate through labels directly
    for label in y_train_labels:
        if label == 0:  # Assuming 0 is malignant
            malignant_count += 1
        else:  # Assuming any other value is benign
            benign_count += 1


    classes = ['Malignant', 'Benign']
    num = [malignant_count, benign_count]

    plt.bar(classes, num)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.show()



    #### Dummy classifier 
    dummy_classifier_model( x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels)

    #### Random Forest
    random_forest_model( x_features_train, x_features_val, x_features_test, y_train_labels, y_val_labels, y_test_labels)

    #### SVM
    svm_model_task1 ( x_train_data, x_val_data, x_test_data, y_train_labels, y_val_labels, y_test_labels)


    ### CNN
    cnn()


    #################Task 2###############
    print("##################################### Task 2 ###########################")
    train_dataset_t2, val_dataset_t2, test_dataset_t2, y_train_labels_t2, y_val_labels_t2, y_test_labels_t2  = reading_datafile_task2()
    x_train_t2, x_val_t2, x_test_t2, y_train_labels_t2, y_val_labels_t2, y_test_labels_t2 = flatten_dataset(train_dataset_t2, val_dataset_t2, test_dataset_t2, y_train_labels_t2, y_val_labels_t2, y_test_labels_t2)
    x_train_data_t2, x_val_data_t2, x_test_data_t2 = normalise_data(x_train_t2, x_val_t2, x_test_t2)


    # Class distribution 
    class0 =  0
    class1 =  0
    class2 =  0
    class3 =  0
    class4 =  0
    class5 =  0
    class6 =  0
    class7 =  0
    # Iterate through labels directly
    for label in y_train_labels_t2:
        if label == 0:  # Assuming 0 is malignant
            class0 += 1
        elif (label ==1):  # Assuming any other value is benign
            class1 += 1
        elif (label ==2):  # Assuming any other value is benign
            class2 += 1
        elif (label ==3):  # Assuming any other value is benign
            class3 += 1
        elif (label ==4):  # Assuming any other value is benign
            class4 += 1
        elif (label ==5):  # Assuming any other value is benign
            class5 += 1
        elif (label ==6):  # Assuming any other value is benign
            class6 += 1
        elif (label ==7):  # Assuming any other value is benign
            class7 += 1

    classes = ['0', '1', '2', '3', '4', '5', '6', '7']
    num = [class0, class1, class2, class3,class4,class5,class6,class7]

    plt.bar(classes, num)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.show()

    # SVM
    svm_model_task2(x_train_data_t2, x_val_data_t2, x_test_data_t2, y_train_labels_t2, y_val_labels_t2, y_test_labels_t2 )

    # CNN 
    cnn_task2()