# Brainwear-Master-project
Developed a machine learning model to identify walking activity from accelerometer data for gait analysis in brain cancer patients. Utilized a supervised SVM model to differentiate walking from other activities. Experienced in Python, pandas for data analysis, and scikit-learn for machine learning.

SVM Model for Automated Identification of Walking Epochs
This project implements a Support Vector Machine (SVM) model for automating the identification of pure walking epochs from accelerometer data.

# Overview
The program enables automated extraction of pure walking epochs in 30-second intervals, streamlining the process for gait analysis.

# Prerequisites
Timeseries accelerometer data and raw output generated using the UK Biobank program are required to extract walking activity.
## Dataset
Due to its large size, the dataset used for SVM training and testing is not included in this repository.

# Setup
Create a folder named walk_vol to hold extracted walking files from each volunteer.
Ensure that the extracted walking files are placed in the walk_data directory.
Data Processing
Use svm_dataset.py to create a labeled dataset suitable for SVM model training and validation.
The script generates graphs for each 30-second epoch and prompts user input to label epochs as pure walking (1) or mixed activity (0).
## SVM Model
Hyperparameters can be tuned using GridsearchCV with 5-fold cross-validation at the end of the code.
After determining the best parameters and kernel, update the model for training on the train set using svm_walk.py.
The model's performance, including a confusion matrix and score metrics, is provided on the test set.
## Extras
splitdata.py: Chunk large CSV files into smaller ones to mitigate memory issues.
extract_w.py: Extract walking from individual raw files and merge them.
learning.py: Generate a learning curve to assess model performance with different sample sizes.
gridplot.py: Produce a heatmap of SVM scores for various parameter combinations.
# Contact
For questions or feedback, feel free to reach out at fatima.emb95@gmail.com.
