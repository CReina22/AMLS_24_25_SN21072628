# AMLS_assignment24_25

## Description
Task 1 (Binary Classification) - Used the BreastMNIST images to train machine learning and deep learning model to classify images into malignant(cancerous) and benign/normal (non-cancerous) classes. Dataset consists of 780 28x28 greyscale images, which were pre-split into train (546 images) , validation (78 images) and testing (156 images) set. Models tested were Random Forest (RF), Dummy classifier, Support Vector Machine (SVM) and Convolutional Neural Network (CNN).

Task 2 (Multi-class Classification) - Used the BloodMIST images to train SVM and CNN model to classify human blood cells images into 8 classes (basophil (0), eosinophil (1), erythroblast (2), immature granulocytes (3), lymphocyte (4), monocyte (5), neutro-phil (6), and platelet (7)). The data was pre-split into training (11,959 images), validation (1,715 images), and testing (3,421 images) sets. 


## Role for each file 
| File name                   | Function                                                                       |
|:----------------------------|:-------------------------------------------------------------------------------|
| dummy_classifier.py         | code for training, validating and testing dummy classifier (Task 1)            |
| random_forest.py            | code for training, validating and testing RF model   (Task 1)                  | 
| svm_task1.py                | code for training, validating and testing SVM model  (Task 1)                  | 
| cnn_task1.py                | code for training, validating and testing CNN model   (Task 1)                 |
| svm_task2.py                | code for training, validating and testing SVM model  (Task 2)                  | 
| CNN_task2.py                | code for training, validating and testing CNN model  (Task 2)                  |
| main.py                     | code for preprocessing of data and calling the ML and DL model  (Task 2)       |

## Packages
scikit-learn
numpy
scikit-image
matplotlib
scikit-learn
seaborn
torchvision
medmnist
tqdm

## To successfully run the code
1. Create and activate a virtual environment
    - To create a virtual environment:
        Unix/macOS: `python3 -m venv .venv`
        Windows: `py -m venv .venv`
    - To activate the virtual environment:
    Unix/macOS: `source .venv/bin/activate`
    Windows: `.venv\Scripts\activate`
2. Install the requirement which consists of all libraries
    - `pip install -r requirements.txt`
3. Type `python main.py` in terminal to run code