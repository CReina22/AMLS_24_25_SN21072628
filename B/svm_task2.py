from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay,classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt




def svm_model_task2( x_train_task2, x_val_task2, x_test_task2, y_train_labels_task2, y_val_labels_task2, y_test_labels_task2):
    """ Performs training, cross validation and grid search to find best hyperparameters
    and testing. 


     Parameters
     x_train_task2: normalised training dataset
     x_val_task2: normalised validation dataset
     x_test_task2: normalised testing dataset
     y_train_labels_task2: flattened training labels
     y_val_labels_task2: flattened validation labels
     y_test_labels_task2: flattened testing labels

     """


    print("**************************************Results from SVM *****************************************")
    print("Initial results")
    
    svc = SVC(random_state=20 ,class_weight='balanced')   # Create the SVM Classifier
    svc.fit(x_train_task2, y_train_labels_task2)

    # Predict using the trained model
    y_pred = svc.predict(x_test_task2)

    # Calculate various metrics
    accuracy = accuracy_score(y_test_labels_task2, y_pred)
    # Print the metrics
    print("Accuracy:", round(accuracy, 2))

    # Plot confusion matrix
    s = sns.heatmap(confusion_matrix(y_test_labels_task2, y_pred), annot=True, fmt='.0f')
    s.set(xlabel="Predicted Labels", ylabel="True Labels")
    plt.title('Confusion Matrix SVM initial')
    plt.show()

    # Print classification report 
    print("Classification Report:")
    print(classification_report(y_test_labels_task2, y_pred))

    #hyperparameter tuning                                    (Hyperparameter tuning section commented out due to long exceution time)
    # print("Hyperparameter tuning (Grid Search):")

    ## Hyperparameter tuning ()
    #param_dist = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     #         'decision_function_shape': ['ovo', 'ovr'],
      #        'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
       #      'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}


    # Create a SVMt classifier
    #svc_hypertuning = SVC(class_weight='balanced',random_state = 20)

    # Use random search to find the best hyperparameters
    #grid_search = GridSearchCV(svc_hypertuning , 
                   #        param_grid = param_dist,  
                   #         cv=5)

    # Fit the random search object to the data
    #grid_search.fit(x_val_task2, y_val_labels_task2)  
    # Print the best hyperparameters
    #print('Best hyperparameters for SVM:',  grid_search.best_params_)

    #### Testing model
    print("Testing results")
    svc_best_model = SVC(kernel =  'rbf',decision_function_shape =  'ovo', C = 10, gamma = 0.01, probability=True, random_state = 20,class_weight='balanced')
    svc_best_model.fit(x_train_task2, y_train_labels_task2)

    y_pred_hyper = svc_best_model.predict(x_test_task2)

    accuracy = accuracy_score(y_test_labels_task2,   y_pred_hyper)

    print("Accuracy:", round(accuracy,2))

    # Plot confusion matrix
    s = sns.heatmap(confusion_matrix(y_test_labels_task2,  y_pred_hyper), annot=True, fmt='.0f')
    s.set(xlabel="Predicted Labels", ylabel="True Labels")
    plt.title('Confusion Matrix SVM')
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test_labels_task2, y_pred_hyper))


