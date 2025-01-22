#Preprocessing
import torchvision.transforms as transforms
from medmnist import BreastMNIST
import random

# For model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# for testing and training
from sklearn.utils import class_weight
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc
import seaborn as sns
import copy
from tqdm import tqdm

# for plotting
import matplotlib.pyplot as plt

class CNNNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNNNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 16),  # Hidden layer
            nn.ReLU(),  
            nn.Dropout(0.4),                                # ReLU activation
            nn.Linear(16, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def train_model (model, criterion, optimizer, train_loader, val_loader, num_epochs, scheduler):
    # train
    train_total_loss = []           # stores the avg loss per epoch
    train_accuracies = []           # stores the avg accuracy per epoch
    # validation
    val_total_loss = []             # stores the avg loss per epoch
    val_accuracies = []             # stores the avg accuracy per epoch

    best_val_loss = float('inf')
    early_stopping = 0

    # for early stopping
    patience = 7  # Number of epochs with no improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement
    patience_counter = 0

    for epoch in range(num_epochs):
        y_train_true = torch.tensor([])         # stores the actual label
        y_train_score = torch.tensor([])        # stores predicted label
        losses = 0                              # stores the loss 

        model.train()                           # sets model into training mode
        for inputs, targets in tqdm(train_loader):      
            targets = targets.view(-1, 1).float()
            optimizer.zero_grad()               # resets gradient
            outputs = model(inputs)             # output from the CNN model

            loss = criterion(outputs, targets)      # computes the loss 
            loss.backward()                         # computes gradient
            optimizer.step()                        # adjusts the weights and biases of the model 
            
            losses += loss.item()                   # Add the loss for each instance
            proba = torch.sigmoid(outputs)          # covert raw output to proabability
            y_pred = proba.round()                  # round probability to 1 or 0

            y_train_true = torch.cat(( y_train_true, targets), 0)       # joins the actual labels from dataset 
            y_train_score = torch.cat((y_train_score, y_pred), 0)       # joins the predicted labels from dataset 

        y_train_true = y_train_true.numpy()             # converts all actual label tensor to numpy array
        y_train_score = y_train_score.detach().numpy()  # conerts all predicted label tensor to numpy array
        # Accuracy per epoch
        train_accuracy = accuracy_score(y_train_true,y_train_score)     # calculates accuracy
        train_accuracies.append(train_accuracy)                         # adds the accuracy to the list

        # Average loss for the epoch
        avg_loss = losses / len(train_loader)       # averages the loss over the number of batches
        train_total_loss.append(avg_loss)    

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.2f}, Training Accuracy: {train_accuracy:.2f}")
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode

        y_val_true = torch.tensor([])               # stores the actual label
        y_val_score = torch.tensor([])              # stores the avg accuracy per epoch
    
        val_losses = 0                              # stores the loss calc during an epoch

        with torch.no_grad():                       # disable gradient tracking
            for inputs, targets in val_loader:      
                outputs = model(inputs)                     # output from the CNN model
                targets = targets.view(-1, 1).float()       
                loss = criterion(outputs, targets)          #  computes the loss 
                val_losses += loss.item()                   #  Add the loss for each instance

                val_proba = torch.sigmoid(outputs)          # covert raw output to proabability
                y_val_pred = val_proba.round()              # round probability to 1 or 0

                y_val_true = torch.cat(( y_val_true, targets), 0)       #  joins the actual labels from dataset 
                y_val_score = torch.cat((y_val_score, y_val_pred), 0)   # joins the predicted labels from dataset 

            y_val_true = y_val_true.numpy()                             #  converts all actual label tensor to numpy array
            y_val_score = y_val_score.detach().numpy()                  # conerts all predicted label tensor to numpy array
            
            val_accuracy = accuracy_score(y_val_true,y_val_score)       #  calculates accuracy
            val_accuracies.append(val_accuracy)                         #  adds the accuracy to the list

            # Average loss for the epoch
            avg_val_loss = val_losses / len(val_loader)                 # averages the loss over the number of batches
            val_total_loss.append(avg_val_loss)   


            print(f"Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}") 

            if avg_val_loss < best_val_loss - min_delta: 
                if   early_stopping == 0:
                    best_val_loss = avg_val_loss        
                    best_model_weights = copy.deepcopy(model.state_dict())              # stores the model
                    patience_counter = 0                                                # Reset the patience counter if we have improvement
            else:
                patience_counter += 1
                if patience_counter >= patience:
                     if early_stopping == 0:
                        early_stopping = 1
                        stop_epoch_num = epoch + 1
                        print(f"Early stopping at epoch {stop_epoch_num}")

        scheduler.step()
                    
    torch.save(best_model_weights, 'best_model.pth')                        # saves the model
    plotting_graph(val_accuracies, val_total_loss,train_accuracies, train_total_loss, stop_epoch_num)

    #return  val_accuracies, val_total_loss,train_accuracies, train_total_loss


def test_model(model, criterion, test_loader):
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))             # Loads the model
    model.eval()  # Set the model to evaluation mode                # set to evaluation mode

    y_test_true = torch.tensor([])                          #  stores the actual label
    y_test_score = torch.tensor([])                         # stores the avg accuracy per epoc
    y_test_prob = torch.tensor([])
    test_losses = 0

    with torch.no_grad():
        for inputs, targets in test_loader:                      
            outputs = model(inputs)                     # obtains result from model
            targets = targets.view(-1, 1).float()       
            loss = criterion(outputs, targets)          #  computes the loss 
            test_losses += loss.item()                  #  Add the loss for each instance
    
            test_proba = torch.sigmoid(outputs)         #  covert raw output to proabability
            y_test_pred = test_proba.round()            #  round probability to 1 or 0

            y_test_true = torch.cat(( y_test_true, targets), 0)             #  joins the actual labels from dataset 
            y_test_score = torch.cat((y_test_score, y_test_pred), 0)        # joins the predicted labels from datase#
            y_test_prob = torch.cat((y_test_prob, test_proba), 0)
            
            
        y_test_true = y_test_true.detach().numpy()                               #  converts all actual label tensor to numpy array
        y_test_score = y_test_score.detach().numpy()                    # conerts all predicted label tensor to numpy array
        y_test_prob = y_test_prob.detach().numpy()
        # Accuracy per epoch
        test_accuracy = accuracy_score(y_test_true,y_test_score)        #  calculates accuracy

        avg_val_loss = test_losses / len(test_loader)                   ## averages the loss over the number of batches

        print(f" Testing Accuracy: {test_accuracy:.2f}")
        print(f" Testing Loss: {avg_val_loss:.2f}")

        con_matrix = confusion_matrix(y_test_true, y_test_score) 

        plt.figure(figsize=(6, 4))
        sns.heatmap(con_matrix, annot=True, fmt='.0f', cmap='coolwarm', center=0, linewidths=0.5)
        plt.title('Confusion Matrix')
        plt.show()
        print(con_matrix)


        # Compute ROC curve
        fpr, tpr, _ = roc_curve( y_test_true,  y_test_prob)
  
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Diagonal line (no discrimination)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return test_accuracy , avg_val_loss   

def plotting_graph (val_accuracies, val_losses, training_accuracies, training_losses,stopping_epoch):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(val_accuracies)), val_accuracies, label="Validation Accuracy",color="orange")
    plt.plot(range(len(training_accuracies)), training_accuracies, label="Training Accuracy",color="blue")
    plt.axvline(stopping_epoch, color = 'r', label = 'Early stopping', linestyle='dashed')
    #plt.plot(range(len(val_accuracies[1])), val_accuracies[1], label="SGD",color="red")
    #plt.plot(range(len(val_accuracies[2])), val_accuracies[2], label="RMSProp",color="green")
    #plt.plot(range(len(val_accuracies[3])), val_accuracies[3], label="Adagrad",color="purple")
   # plt.plot(range(len(val_accuracies[4])), val_accuracies[4], label="256",color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy Over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(range(len(val_losses)),val_losses,label="Validation Loss",color="orange")
    plt.plot(range(len(training_losses)),training_losses,label="Training Loss",color="blue")
    plt.axvline(stopping_epoch, color = 'r', label = 'Early stopping', linestyle='dashed')
    #plt.plot(range(len(val_losses[1])),val_losses[1],label="SGD",color="red")
    ##plt.plot(range(len(val_losses[2])),val_losses[2],label="RMSProp",color="green")
    #plt.plot(range(len(val_losses[3])),val_losses[3],label="Adagrad",color="purple")
   # plt.plot(range(len(val_losses[4])),val_losses[4],label="256",color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Loss Over Epochs")
    plt.show()



def cnn():
    random.seed(20)                         # Set the random seed for Python's random module
    np.random.seed(20)                      # Set the random seed for NumPy
    torch.manual_seed(20)                   # Set the random seed for PyTorch CPU operations

    #epochs = [25, 50, 75, 100]
    #BATCH_SIZE = [16, 32, 64, 128, 256]
    #learning_rate = [0.1, 0.01, 0.001, 0.0001]

    #optimizers = ["Adam", 'SGD', 'RMSProp', 'Adagrad']
    optimiser = 'Adam'
    
    epoch = 100
    BATCH_SIZE = 32
    learning_rate = 0.0001
    
    #train_accuracies_all = np.zeros((len(BATCH_SIZE), epoch))
    #train_losses_all = np.zeros((len(BATCH_SIZE), epoch))
  
    #val_accuracies_all = np.zeros((len(BATCH_SIZE), epoch))
    #val_losses_all = np.zeros((len(BATCH_SIZE), epoch))

    #val_accuracies_all = np.zeros((len(optimizers), epoch))
    #val_losses_all = np.zeros((len(optimizers), epoch))
    val_accuracies_all = []
    val_losses_all = []

    training_accuracies_all = []
    training_losses_all = []


    test_acc_all = []
    test_loss_all = []

    acc_avg = []
    loss_avg = []

    data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
    ])

    transform= transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

        # load the data
  
    train_dataset = BreastMNIST(split='train', transform=transform)
    test_dataset = BreastMNIST(split='test', transform=data_transform)
    val_dataset = BreastMNIST(split='val', transform=data_transform)
    
   
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    model = CNNNet(in_channels=1, num_classes=1)

    class_weights=class_weight.compute_class_weight('balanced', 
                                                classes=np.unique(train_dataset.labels[:,0]), 
                                                y=train_dataset.labels[:,0])

    criterion = nn.BCEWithLogitsLoss(torch.tensor([class_weights[1] / class_weights[0]]))
    #i = 0
    

    if optimiser == 'Adam': 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimiser == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimiser == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size= 25, gamma=0.1)

    #val_accuracies_all[0], val_losses_all[0], training_accuracies_all[0], training_losses_all[0] = train_model (model, criterion, optimizer, train_loader, val_loader, epoch)
    train_model (model, criterion, optimizer, train_loader, val_loader, epoch, scheduler)
    
    #i+=1
    accuracy, losses = test_model(model, criterion, test_loader)
    test_acc_all.append(accuracy)
    test_loss_all.append(losses)
            
    for i in range(len(test_acc_all)):
        print(round(test_acc_all[i],2), round(test_loss_all[i],2))
 

    #plotting_graph (val_accuracies_all, val_losses_all)
    

