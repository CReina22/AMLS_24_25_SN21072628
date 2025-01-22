# AMLS_assignment24_25

## Description
Task 1 (Binary Classification) - The BreastMNIST dataset was used to train machine learning and deep learning models to classify images into malignant (cancerous) and benign/normal (non-cancerous) categories. The dataset consists of 780 28x28 grayscale images, pre-split into training (546 images), validation (78 images), and testing (156 images) sets. The models tested included Random Forest (RF), Dummy Classifier, Support Vector Machine (SVM), and Convolutional Neural Network (CNN).

Task 2 (Multi-class Classification) - The BreastMNIST dataset was used to train machine learning and deep learning models to classify images into malignant (cancerous) and benign/normal (non-cancerous) categories. The dataset consists of 780 28x28 grayscale images, pre-split into training (546 images), validation (78 images), and testing (156 images) sets. The models tested included Random Forest (RF), Dummy Classifier, Support Vector Machine (SVM), and Convolutional Neural Network (CNN).




## Role for each file 
| File name                   | Function                                                                       |
|:----------------------------|:-------------------------------------------------------------------------------|
| dummy_classifier.py         | Code for training, validating and testing dummy classifier (Task 1)            |
| random_forest.py            | Code for training, validating and testing RF model   (Task 1)                  | 
| svm_task1.py                | Code for training, validating and testing SVM model  (Task 1)                  | 
| cnn_task1.py                | Code for training, validating and testing CNN model   (Task 1)                 |
| svm_task2.py                | Code for training, validating and testing SVM model  (Task 2)                  | 
| CNN_task2.py                | Code for training, validating and testing CNN model  (Task 2)                  |
| main.py                     | Code for preprocessing of data and calling the ML and DL models                |
| requirments.txt             | Contains all libraries needed to be installed                                  |

## Packages
- scikit-learn
- numpy
- scikit-image
- matplotlib
- scikit-learn
- seaborn
- torchvision
- medmnist
- tqdm
- torch

## To successfully run the code
1. Clone the repository
2. Create and activate a virtual environment
    - To create a virtual environment:
        Unix/macOS: `python3 -m venv .venv`
        Windows: `py -m venv .venv`
    - To activate the virtual environment:
    Unix/macOS: `source .venv/bin/activate`
    Windows: `.venv\Scripts\activate`
3. Install the requirement which consists of all libraries
    - `pip install -r requirements.txt`
4. Type `python main.py` in terminal to run code
