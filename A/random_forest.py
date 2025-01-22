from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay,classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def random_forest_model(x_train, x_val, x_test, y_train_labels, y_val_labels, y_test_labels):
    print("Random Forest intial results")
    rf = RandomForestClassifier(random_state= 20,class_weight='balanced')
    rf.fit(x_train, y_train_labels)

    y_pred = rf.predict(x_test)

    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred)
    recall = recall_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred)
    print("Accuracy:", round(accuracy,2))
    print("Precision:", round(precision,2))
    print("Recall:", round(recall,2))
    print("F1 Score:", round(f1,2))

    # Plot confusion matrix
    con_matrix = sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True,fmt='.0f')
    con_matrix.set(xlabel="Predicted Labels", ylabel="True Labels")
    plt.title('Confusion Matrix Random Forest initial')
    plt.show()

    # Print classification report with zero_division parameter correctly used
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred))

    #############################################hyperparameter tuning
    print("Hyperparameter tuning (Grid Search):")


    ## Hyperparameter tuning ()
    param_dist = {'n_estimators': [100, 150, 200],
                'max_depth': [10, 20, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5, 10],
                'max_leaf_nodes': [10, 20, 30]}

    # Create a random forest classifier
    rf = RandomForestClassifier(class_weight='balanced',random_state= 20)

    # Use random search to find the best hyperparameters
    grid_search_rf = GridSearchCV(rf, 
                                param_grid = param_dist,  
                                cv=5)

    # Fit the random search object to the data
    grid_search_rf.fit(x_val, y_val_labels)  #13mins 28.9s

    # Print the best hyperparameters
    print('Best hyperparameters for Random Forest:',  grid_search_rf.best_params_)

    ####################################### Testing model
    print("Random Forest Testing")
    grid_search_rf.fit(x_train, y_train_labels)

    y_pred = grid_search_rf.predict(x_test)

    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred)
    recall = recall_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred)
    print("Accuracy:", round(accuracy,2))
    print("Precision:", round(precision,2))
    print("Recall:", round(recall,2))
    print("F1 Score:", round(f1,2))

    # Plot confusion matrix
    con_matrix = sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt='.0f')
    con_matrix.set(xlabel="Predicted Labels", ylabel="True Labels")
    plt.title('Confusion Matrix  Random Forest')
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred))
