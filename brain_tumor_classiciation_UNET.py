import os
import sys
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from unet import UNet
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the output file path
output_file = 'brain_tumor_classiciation_UNET.txt'

# Redirect standard output to the output file
sys.stdout = open(output_file, 'w')

# Check if CUDA GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Function to save training checkpoint - saved after each epoch is run
def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1,  # Saving next epoch to start from
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

# Function to get a subset of dataset indices - In case it is taking too long to train in the cluster
def get_subset_indices(dataset, fraction=1):
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return indices

# Function to calculate accuracy per class
def accuracy_per_class(conf_matrix):
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    return class_acc

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, classes, filename=None):
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j+0.5, i+0.5, conf_matrix[i, j], ha='center', va='center', color='black')

    if filename:
        plt.savefig(filename)  # Save the plot as an image
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()

# Define the root directory containing the "brain tumor" and "healthy" folders
root_dir = r"C:\Users\Gigabyte\Downloads\Brain Tumor Data Set\Brain Tumor Data Set"

# Transform dataset input to match inputs for ResNet-101 model
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
full_dataset = ImageFolder(root=root_dir, transform=data_transform)

# Get subset indices
subset_indices = get_subset_indices(full_dataset)

# Create subset dataset
subset_dataset = Subset(full_dataset, subset_indices)

# Split dataset into train, validation, and test sets
num_samples = len(subset_dataset)
train_size = int(0.7 * num_samples)  
val_size = int(0.15 * num_samples)   
test_size = num_samples - train_size - val_size  

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size, test_size])

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the UNet model
model = UNet() 

# Move the model to the appropriate device (CPU or GPU)
model = model.to(device)

# Define a loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Training loop with early stopping based on time limit
num_epochs = 50
start_time = time.time()
time_limit = int(os.getenv('JOB_TIME_LIMIT', '7200000'))
safe_margin = 300

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    for images, labels in train_dataloader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    class_acc = accuracy_per_class(conf_matrix)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_dataloader):.4f}, '
        f'Validation Accuracy: {(correct/total)*100:.2f}%, '
        f'Accuracy per Class: {class_acc}, '
        f'Precision: {precision:.4f}, '
        f'Recall: {recall:.4f}, '
        f'F1 Score: {f1:.4f}')

    # Save Confusion Matrix as Image after the last epoch
    if epoch == num_epochs - 1:
        plot_confusion_matrix(conf_matrix, classes=[str(i) for i in range(2)], filename='confusion_matrix_net.png')

    # Save checkpoint after each epoch
    #save_checkpoint(model, optimizer, epoch, filename=f'checkpoint_epoch_{epoch}.pth.tar')

    # Check for early stopping due to time limit
    epoch_duration = time.time() - epoch_start_time
    time_left = time_limit - (time.time() - start_time)

    if time_left < epoch_duration + safe_margin:
        print(f'Breaking the training loop due to time limit. Time left: {time_left:.2f} seconds.')
        break

# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
all_predicted_labels = []
all_true_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()

        all_predicted_labels.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
test_conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
test_class_acc = accuracy_per_class(test_conf_matrix)

# Calculate precision, recall, and F1 score for test set
test_precision = precision_score(all_true_labels, all_predicted_labels)
test_recall = recall_score(all_true_labels, all_predicted_labels)
test_f1 = f1_score(all_true_labels, all_predicted_labels)


print(f'Test Loss: {test_loss/len(test_dataloader):.4f}, '
    f'Test Accuracy: {(correct/total)*100:.2f}%, '
    f'Accuracy per Class: {test_class_acc}, '
    f'Precision: {test_precision:.4f}, '
    f'Recall: {test_recall:.4f}, '
    f'F1 Score: {test_f1:.4f}')
