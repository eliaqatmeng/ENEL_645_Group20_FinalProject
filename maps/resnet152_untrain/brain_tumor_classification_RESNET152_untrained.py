import os
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet152
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import torch.nn.functional as F

# Define the output file path
output_file = 'maps/resnet152_untrain/brain_tumor_classification_RESNET152_untrained.txt'

# Redirect standard output to the output file
sys.stdout = open(output_file, 'w')


# Check if CUDA GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Function to get a subset of dataset indices - In case it is taking too long to train in the cluster
def get_subset_indices(dataset, fraction=1):
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return indices

# Function to calculate accuracy per class
def accuracy_per_class(conf_matrix):
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    return class_acc


def compute_grad_cam(model, img_tensor, target_class=None):
    model.eval()
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()
    
    output = model(img_tensor)
    
    if target_class is None:
        target_class = output.argmax()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get the gradients of the output with respect to the feature maps
    gradients = img_tensor.grad
    
    # Get the activations of the last convolutional layer using the forward hook
    activations = None
    def hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    hook_handle = model.layer4[-1].relu.register_forward_hook(hook)  # Assuming the last convolutional layer is layer4[-1].relu
        
    # Pass a dummy input through the model to trigger the forward hook
    model(img_tensor)
    
    # Remove the forward hook
    hook_handle.remove()
    
    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight the activations by the corresponding gradients
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # Compute the heatmap by averaging the weighted activations along the channels
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # ReLU on the heatmap
    heatmap = F.relu(heatmap)
    
    return heatmap

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

# Load pre-trained ResNet-101 model
model = resnet152(pretrained=False)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one (assuming your dataset has 2 classes)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming 2 output classes

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
# Save ROC curve and AUC data after the last epoch
    if epoch == num_epochs - 1:
        all_probs = []
        all_true_labels = []
        model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

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






# Function to load and preprocess an image
def load_and_preprocess_image(image_path, resize_dim=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, resize_dim)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return image_tensor, image

# List of image paths
image_paths = [
    r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (1413).jpg", 
    r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (1609).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (2406).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (2425).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Not Cancer  (13).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Not Cancer  (52).jpg"]

# Loop over each image
for image_path in image_paths:
    # Load and preprocess the image
    image_tensor, original_image = load_and_preprocess_image(image_path)

    # Get the predicted class label
    with torch.no_grad():
        model.eval()
        outputs = model(image_tensor)
        _, predicted_class_idx = torch.max(outputs, 1)

    # Map the predicted class index to the actual class label
    class_labels = ['Healthy', 'Brain Tumor']  # Replace with your actual class labels
    predicted_class_label = class_labels[predicted_class_idx]

    # Print the predicted class label
    print(f"Predicted Class Label: {predicted_class_label}")

    # Get the Grad-CAM heatmap
    heatmap = compute_grad_cam(model, image_tensor)
    heatmap = heatmap.detach().cpu().numpy()  # Convert the heatmap tensor to a numpy array

    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    heatmap = np.uint8(255 * heatmap / np.max(heatmap))

    # Apply colormap to the heatmap
    heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = heatmap_colormap * 0.5 + original_image * 0.5
    superimposed_img = np.uint8(superimposed_img)

    # Convert the superimposed image to RGB (cv2 reads images in BGR format)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Create the output directory if it doesn't exist
    os.makedirs("maps/resnet152_untrain", exist_ok=True)

    output_image_path = f"maps/resnet152_untrain/{os.path.basename(image_path).split('.')[0]}_gradcam_output.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM image saved as {output_image_path}")


