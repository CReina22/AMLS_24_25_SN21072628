from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay,classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def svm_model_task1( x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels):
   """ Performs training, cross validation and grid search to find best hyperparameters
   and testing. 


   Parameters
   x_train:  training set obtained from normalisation
   x_val: validation set obtained from normalisation
   x_test: testing set obtained from normalisation
   y_train_labels: flattened training labels
   y_val_labels: flattened validation labels
   y_test_labels: flattened testing labels

   """

   print("**************************************Results from SVM *****************************************")
   print("Initial results")
   
   svc = SVC(random_state=20 ,class_weight='balanced')  # Create a SVM classifier
   svc.fit(x_train, y_train_labels)

   # Predict using the trained model
   y_pred = svc.predict(x_test)

   # Calculate various metrics
   accuracy = accuracy_score(y_test_labels, y_pred)
   precision = precision_score(y_test_labels, y_pred, zero_division=0)
   recall = recall_score(y_test_labels, y_pred)
   f1 = f1_score(y_test_labels, y_pred)

   # Print the metrics
   print("Accuracy:", round(accuracy, 2))
   print("Precision:", round(precision, 2))
   print("Recall:", round(recall, 2))
   print("F1 Score:", round(f1, 2))

   # Plot confusion matrix
   s = sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt='.0f')
   s.set(xlabel="Predicted Labels", ylabel="True Labels")
   plt.title('Confusion Matrix SVM initial')
   plt.show()

   # Print classification report with zero_division parameter correctly used
   print("Classification Report:")
   print(classification_report(y_test_labels, y_pred))

   #hyperparameter tuning
   print("Hyperparameter tuning (Grid Search):")

   ## Hyperparameter tuning 
   param_dist = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}


   # Create a SVM classifier
   svc_hypertuning = SVC(class_weight='balanced',random_state = 20)

   # Use random search to find the best hyperparameters
   grid_search = GridSearchCV(svc_hypertuning , 
                           param_grid = param_dist,  
                           cv=5)

   # Fit the random search object to the data
   grid_search.fit(x_val, y_val_labels)  
   # Print the best hyperparameters
   print('Best hyperparameters for SVM:',  grid_search.best_params_)

   #### Testing model
   print("Testing results")
   grid_search.fit(x_train, y_train_labels)    # Train model

   y_pred = grid_search.predict(x_test)    # testing model

   # Calculate various metrics
   accuracy = accuracy_score(y_test_labels, y_pred)
   precision = precision_score(y_test_labels, y_pred)
   recall = recall_score(y_test_labels, y_pred)
   f1 = f1_score(y_test_labels, y_pred)
   print("Accuracy:", round(accuracy,2))
   print("Precision:", round(precision,2))
   print("Recall:", round(recall,2))
   print("F1 Score:", round(f1,2))

   # Plot confusion matrix
   s = sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt='.0f')
   s.set(xlabel="Predicted Labels", ylabel="True Labels")
   plt.title('Confusion Matrix SVM')
   plt.show()

   print("Classification Report:")
   print(classification_report(y_test_labels, y_pred))
