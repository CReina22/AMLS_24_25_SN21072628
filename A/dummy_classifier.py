from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay,classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

def dummy_classifier_model( x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels):
   """ Performs training, cross validation and grid search to find best hyperparameters
    and testing. 


     Parameters
     x_train: raw training dataset
     x_val: raw validation dataset
     x_test: raw testing dataset
     y_train_labels: flattened training labels
     y_val_labels: flattened validation labels
     y_test_labels: flattened testing labels

     """
   
   print("**************************************Results from dummy classifier *****************************************")
   print("Initial results")
   # Fit the Dummy Classifier
   model_dummy = DummyClassifier(random_state=20)
   model_dummy.fit(x_train, y_train_labels)

   # Predict using the trained model
   y_pred = model_dummy.predict(x_test)

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
   plt.title('Confusion Matrix')
   plt.show()

   # Print classification report with zero_division parameter correctly used
   print("Classification Report:")
   print(classification_report(y_test_labels, y_pred, zero_division=0))

   #hyperparameter tuning
   print("Hyperparameter tuning (Grid Search):")

   ## Hyperparameter tuning 
   param_dist = {'strategy': ['most_frequent', 'prior', 'stratified', 'uniform', 'constant'],
               'constant' : [0, 1]}

   # Create a random forest classifier
   dummy = DummyClassifier()

   # Use random search to find the best hyperparameters
   grid_search = GridSearchCV(dummy, 
                              param_grid = param_dist,  
                              cv=5)

   # Fit the random search object to the data
   grid_search.fit(x_val, y_val_labels)  
   # Print the best hyperparameters
   print('Best hyperparameters for Dummy classifier:',  grid_search.best_params_)

   #### Testing model
   print("Testing results:")
   grid_search.fit(x_train, y_train_labels)

   y_pred = grid_search.predict(x_test)

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
   plt.title('Confusion Matrix')
   plt.show()