DRIVE LINK: https://drive.google.com/drive/folders/1TWjjQrsCVBKhvx8TEP7pBsZK9zaMsx3i?usp=drive_link


## Image Classification using Sum-Difference Histogram-based Texture Features and Comparison with GLCM Features

### Group Members

- Akshat Kumar (22B4513)
- Vishal Gautam (22B0065)
- Alvin Bunny Simpson (22B0015)

### Introduction

This project focuses on classifying images using Sum-Difference Histogram-based texture features and evaluating the classification performance in comparison with Gray-Level Co-occurrence Matrix (GLCM) features. We employ Random Forest (RF) and Artificial Neural Network (ANN) classifiers for this task.

Image classification plays a pivotal role in various domains including medical imaging, remote sensing, and surveillance, aiding tasks such as disease diagnosis, environmental monitoring, and security.

### Dataset
The dataset 'Image.rar' consists of images from 7 distinct classes: agriculture, baseballdiamond, beach, buildings, denseresidential, forest, and harbor. Each class contains 500 images.

### Steps to Reproduce Results

1. Download and Extract the 'images.rar' file from DRIVE LINK and specify the directory path in the `get_images()` function.
2. Replace 'MAIN_DIRECTORY' with the location of the images folder data.
3. Ensure that the Jupyter notebook is in the same directory as the images folder.

I have already extracted the GLCM and SDH features from all the images and saved them in two CSV files, which are hosted on Google Drive. You can download these files directly by running the notebook.




### 1. Using Sum-Difference Histogram-based Texture Features
We extract Sum-Difference Histogram-based texture features from each image in the 'Image.rar' dataset. By utilizing the `get_image(k, i)` function, we obtain the grayscale array corresponding to the ith image saved in the kth folder (class). These features are computed using the `compute_SD_histogram(grayscale_array)` function, resulting in an array of features ([energy, entropy, contrast, correlation, dissimilarity]). Subsequently, a new dataset is created using the `Calculate_SDH_Values(Main_Dataset)` function, containing image numbers, class labels, and the computed texture features. We then utilize this dataset as input for ANN and RF classifiers to predict the class of an image based on its texture features.

### 2. Using GLCM Features
In addition to Sum-Difference Histogram-based texture features, we extract GLCM features from each image. A new dataset is created similar to the Sum-Difference Histogram-based method, and ANN and RF classifiers are trained on this dataset to predict the class of an image based on its GLCM texture features.

### Results
The classification results, including accuracy, precision, recall, and F1-score, are presented for each classifier using Sum-Difference Histogram-based texture features and GLCM features.

### Conclusion
This project highlights the effectiveness of Sum-Difference Histogram-based texture features for image classification tasks and compares their performance with GLCM features. The results underscore the significance of feature selection and extraction methods in achieving accurate classification outcomes.