#!/usr/bin/env python
# coding: utf-8

# ## LINK TO GOOGLE DRIVE(DATA+IPYNB): https://drive.google.com/drive/folders/1TWjjQrsCVBKhvx8TEP7pBsZK9zaMsx3i?usp=sharing

# In[167]:


import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import re
import matplotlib.image as mpimg


# In[168]:


from skimage.feature import graycomatrix , graycomatrix 
from skimage import io, color
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import random
from PIL import Image


# # Accessing Images
# 
# To access the images, please provide the location of the main directory where the images are saved.
# 
# ## Expected Data Structure
# 
# The main directory should point to where the 'images' folder is saved. Within the 'images' folder, there should be seven subfolders, each containing images corresponding to a specific class.
# 
# ### Link to Dataset
# [Download Dataset](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification?resource=download)
# 
# In this dataset, we have selected images only from the following classes: 'agricultural', 'baseballdiamond', 'beach', 'buildings', 'denseresidential', 'forest', 'harbor'. Therefore, please ensure that these folders are present within the 'images' folder.
# 

# # Creating GLCM and SDH Dataset

# In[169]:


def get_image(k,i):
    main_directory = r'C:\Users\Akshat Kumar\OneDrive - Indian Institute of Technology Bombay\Desktop\Final_Dataset\Images'
    # Get the list of folders inside the 'Images' folder
    folders = os.listdir(main_directory)

    # Define the path to the selected folder
    selected_folder_path = os.path.join(main_directory, folders[k])

    # Get the list of image files in the selected folder
    image_files = os.listdir(selected_folder_path)

    # Define the path to the selected image
    selected_image_path = os.path.join(selected_folder_path, image_files[i])

    # Load the selected image
    rgb_image = Image.open(selected_image_path)

    # Convert the image to grayscale
    grayscale_image = rgb_image.convert('L')

    # Convert the grayscale image to an array
    grayscale_array = np.array(grayscale_image)
    return grayscale_array

plt.imshow(get_image(2,499))


# ## Calulating The GLCM Features 

# In[170]:


import numpy as np
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_features(image_label, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True):
    """
    Calculate GLCM features for a given labeled image.
    
    Parameters:
        image_label (2D array): Labeled image.
        distances (list of int, optional): List of pixel pair distances. Default is [5].
        angles (list of float, optional): List of pixel pair angles in radians. Default is [0, np.pi/4, np.pi/2, 3*np.pi/4].
        levels (int, optional): Number of gray levels. Default is 256.
        symmetric (bool, optional): Whether the GLCM is symmetric. Default is True.
        normed (bool, optional): Whether to normalize the GLCM. Default is True.
        
    Returns:
        list: List of GLCM features calculated for the input image.
    """
    # Calculate GLCM
    glcm = graycomatrix(image_label, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    
    # Define GLCM properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    # Calculate GLCM properties
    glcm_features = np.zeros(len(properties))
    for i, prop in enumerate(properties):
        glcm_features[i] = np.mean(graycoprops(glcm, prop))
        
    return glcm_features


# In[171]:


image = get_image(0,0)
plt.imshow(image)


# ## Calulating The SDH Features

# In[172]:


def compute_SD_histogram(image):
    # Convert the image to grayscale
    gray = image
    
    # Define the parameters for the Sum-Difference histogram
    num_bins = 256
    sum_diff_hist = np.zeros((num_bins, num_bins), dtype=np.float32)
    
    # Compute the Sum-Difference histogram
    rows, cols = gray.shape
    for i in range(1, rows):
        for j in range(1, cols):
            sum_val = int(gray[i, j]) + int(gray[i-1, j-1])
            diff_val = int(gray[i, j]) - int(gray[i-1, j-1])
            if sum_val < num_bins and diff_val < num_bins:
                sum_diff_hist[sum_val, diff_val] += 1
    
    # Normalize the histogram
    sum_diff_hist /= np.sum(sum_diff_hist)
    
    return sum_diff_hist

def compute_texture_features(texture_hist):
    # Compute features from the texture histogram
    energy = np.sum(texture_hist ** 2)  # Energy (Uniformity)
    entropy = -np.sum(texture_hist * np.log(texture_hist + 1e-10))  # Entropy
    contrast = np.sum((np.arange(texture_hist.shape[0]) - np.mean(texture_hist)) ** 2)  # Contrast
    correlation = np.sum((np.arange(texture_hist.shape[0])[:, None] * np.arange(texture_hist.shape[1])) * texture_hist) / (np.sqrt(np.sum(np.arange(texture_hist.shape[0]) ** 2) * np.sum(np.arange(texture_hist.shape[1]) ** 2)))  # Correlation
    dissimilarity = np.sum(np.abs(np.arange(texture_hist.shape[0]) - np.mean(texture_hist)))  # Dissimilarity
    
    # Return the texture features as a NumPy array
    return np.array([energy, entropy, contrast, dissimilarity])

def compute_texture_features_from_image(image):
    # Compute Sum-Difference histogram
    texture_hist = compute_SD_histogram(image)
    
    # Compute texture features from the histogram
    texture_features = compute_texture_features(texture_hist)
    
    return texture_features


# In[173]:


compute_texture_features_from_image(get_image(1,5))


# ## Extracting GLCM and SDH Features from the Images

# In[174]:


glcm_dataset = np.zeros((8*500,7))
sdh_dataset = np.zeros((8*500,6))


# In[175]:


def Calculate_GLCM_Values(Main_Dataset):
    l = 0
    for k in range(8):
        for i in range(500):
            image = get_image(k,i)
            glcm = calculate_glcm_features(image)
            Main_Dataset[l][0] = k
            Main_Dataset[l][1] = i
            Main_Dataset[l][2] =glcm[0]
            Main_Dataset[l][3] =glcm[1]
            Main_Dataset[l][4] =glcm[2]
            Main_Dataset[l][5] =glcm[3]
            Main_Dataset[l][6] =glcm[4]
            l = l+1
    return Main_Dataset     


# In[176]:


glcm_data = Calculate_GLCM_Values(glcm_dataset)


# In[177]:


glcm_data.shape


# In[178]:


def Calulate_SDH_Values(Main_Dataset):
    l = 0
    for k in range(8):
        for i in range(500):
            image = get_image(k,i)
            SDH = compute_texture_features_from_image(image)
            Main_Dataset[l][0] = k
            Main_Dataset[l][1] = i
            Main_Dataset[l][2] =SDH[0]
            Main_Dataset[l][3] =SDH[1]
            Main_Dataset[l][4] =SDH[2]
            Main_Dataset[l][5] =SDH[3]
            l = l+1
    return Main_Dataset


# In[179]:


sdh_data = Calulate_SDH_Values(sdh_dataset)
sdh_data.shape


# In[180]:


df_sdh = pd.DataFrame(sdh_data)
df_glcm = pd.DataFrame(glcm_data)


# In[181]:


print(sdh_data)
print(sdh_data.shape)


# In[182]:


print(glcm_data)
print(glcm_data.shape)


# In[183]:


# Save df_sdh to CSV
df_sdh.to_csv('df_sdh.csv', index=False)

# Save df_glcm to CSV
df_glcm.to_csv('df_glcm.csv', index=False)


# ## Creating a Dataset of SDH and GLCM Features
# 
# By combining the Sum-Difference Histogram (SDH) and Gray-Level Co-occurrence Matrix (GLCM) features of multiple images, we can create a dataset for various applications.
# 
# ## Directly Downloading the Dataset
# 
# To streamline the process, we have preprocessed the data and uploaded it to Google Drive. You can automatically download the dataset using the following code:
# 
# DRIVE: https://drive.google.com/drive/folders/1TWjjQrsCVBKhvx8TEP7pBsZK9zaMsx3i?usp=drive_link

# In[184]:


import gdown

# URLs of the CSV files
df_glcm_url = 'https://drive.google.com/uc?id=1c7Z8AQxzM-wramvXkDlri7pGbPGDOgPM'
df_sdh_url = 'https://drive.google.com/uc?id=1ewOpQlVBNdSc1ipeUtNGidtMjCxQ8_uz'

# Path to save the downloaded files
df_glcm_path = 'df_glcm.csv'
df_sdh_path = 'df_sdh.csv'

# Download df_glcm.csv
gdown.download(df_glcm_url, df_glcm_path, quiet=False)

# Download df_sdh.csv
gdown.download(df_sdh_url, df_sdh_path, quiet=False)


# # Data Preprocessing 

# In[185]:


df_glcm = pd.read_csv('df_glcm.csv')
df_sdh = pd.read_csv('df_sdh.csv')


# In[186]:


df_sdh


# In[187]:


df_glcm


# In[188]:


# Shuffle the rows of df_glcm
df_glcm = df_glcm.sample(frac=1).reset_index(drop=True)

# Shuffle the rows of df_sdh
df_sdh = df_sdh.sample(frac=1).reset_index(drop=True)


# In[189]:


df_glcm


# In[190]:


ground_truth_glcm = df_glcm.iloc[:,0]
ground_truth_sdh = df_sdh.iloc[:,0]


# In[191]:


ground_truth_glcm


# In[192]:


# Drop the first and second columns by index from df_glcm
df_glcm = df_glcm.drop(df_glcm.columns[[0, 1]], axis=1)

# Drop the first and second columns by index from df_sdh
df_sdh = df_sdh.drop(df_sdh.columns[[0, 1]], axis=1)


# # Visualizing The Data

# In[193]:


df_shuffled_glcm = df_glcm
df_shuffled_sdh = df_sdh


# In[194]:


GLCM_features = ['GLCM_1', 'GLCM_2', 'GLCM_3', 'GLCM_4', 'GLCM_5']
df_shuffled_glcm.columns = GLCM_features
# Create subplots
fig, axs = plt.subplots(5, 1, figsize=(8, 20))

# Plot each GLCM feature against labels
for i, feature in enumerate(GLCM_features):
    axs[i].scatter(df_shuffled_glcm[feature], ground_truth_glcm, alpha=0.5)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Labels')

plt.tight_layout()
plt.show()


# In[195]:


SDH_features = ['SDH_1', 'SDH_2', 'SDH_3', 'SDH_4']

df_shuffled_sdh.columns = SDH_features

# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 20))

# Plot each GLCM feature against labels
for i, feature in enumerate(SDH_features):
    axs[i].scatter(df_shuffled_sdh[feature], ground_truth_sdh, alpha=0.5)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Labels')

plt.tight_layout()
plt.show()


# ## Normalizing The Data

# In[196]:


glcm_d = df_shuffled_glcm.values
sdh_d = df_shuffled_sdh.values


# In[197]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to your data and transform it
glcm_data = scaler.fit_transform(glcm_d)
sdh_data = scaler.fit_transform(sdh_d)


# # Creating ML Models For GLCM 

# In[204]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Define the input shape
input_shape = (glcm_data.shape[1],)

# Create the model
glcm_model = Sequential([
    Dense(64, activation='relu', input_shape=input_shape),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming you're doing binary classification
])

# Compile the model with modified settings
learning_rate = 0.1  # Increased learning rate
loss_type = 'mean_squared_error'  # Changed loss function to mean squared error
optimizer = SGD(learning_rate=learning_rate)  # Changed optimizer to SGD with specified learning rate

glcm_model.compile(optimizer=optimizer, loss=loss_type, metrics=['accuracy'])

# Train the model
glcm_model.fit(glcm_data, ground_truth_glcm, epochs=30, batch_size=32, validation_split=0.2)


# In[207]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(glcm_data, ground_truth_glcm, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model on the training data
rf_model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # Creating ML Models For SDH

# In[200]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Split the data
X_train, X_test, y_train, y_test = train_test_split(sdh_data, ground_truth_sdh, test_size=0.2, random_state=42)

# Step 2: Preprocess the data (if necessary)

# Step 3: Define the neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # Corrected input shape to match the number of features
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')  # 8 classes, so the output layer has 8 units and softmax activation
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)  # Adjust epochs and batch_size as needed

# Step 6: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)


# In[208]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)


# In[ ]:





# In[ ]:




