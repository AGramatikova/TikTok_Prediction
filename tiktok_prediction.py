"""
TikTok Video Popularity Prediction Script

This script demonstrates a simple approach to predict whether a TikTok video will be popular.
It uses a dataset (e.g. TikTok Video Performance Dataset) with features such as likes, comments,
shares, views, hashtags and user metrics. The script defines a binary target variable 'is_popular'
based on the median value of likes and trains a Random Forest classifier.

To use this script:
1. Place the dataset CSV file in the same directory as this script and specify its filename below.
2. Adjust the feature columns as appropriate for your dataset.
3. Run the script to train the model and see evaluation metrics.

Note: For a proper project, consider more sophisticated feature engineering (e.g. number of hashtags,
audio features, posting time) and cross-validation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# TODO: specify your dataset filename (ensure it is available in repository)
DATA_FILE = 'tiktok_video_performance.csv'

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Define target variable: popular if likes greater than median of likes
df['is_popular'] = (df['likes'] > df['likes'].median()).astype(int)

# Select features; exclude the target variable and identifiers
# Modify these columns based on actual dataset columns (comments, shares, views, video_length, etc.)
feature_columns = ['comments', 'shares', 'views']

X = df[feature_columns]
y = df['is_popular']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
