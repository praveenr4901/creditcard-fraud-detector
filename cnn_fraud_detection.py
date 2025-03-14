# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv("creditcard.csv")  # Ensure the dataset is present in the directory

# Display available columns
print("Available columns in dataset:", data.columns)

# Ensure the correct target column is used
target_column = "is_fraud"
if target_column not in data.columns:
    raise KeyError(f"Column '{target_column}' not found in the dataset. Please check the dataset structure.")

# Drop non-numeric columns that contain sensitive information
sensitive_columns = [
    'credit_card_number', 'phone_number', 'email_address', 'billing_address', 'shipping_address'
]
data = data.drop(columns=sensitive_columns, errors='ignore')

# Encode categorical columns
categorical_columns = ['country', 'payment_method']
for col in categorical_columns:
    if col in data.columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))

# Separate features and labels
X = data.drop(target_column, axis=1)
y = data[target_column]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape the data for CNN input (n_samples, n_features, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Print the shape of X to verify compatibility
print("Shape of X after reshaping:", X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a CNN model with appropriate pooling layers
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),  # Define input shape explicitly
    Conv1D(filters=32, kernel_size=3, activation='relu', padding="same"),
    MaxPooling1D(pool_size=2, padding="same"),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"),
    MaxPooling1D(pool_size=2, padding="same"),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Print the model summary
model.summary()

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    class_weight={0: 1, 1: 10}  # Adjust class weights to handle imbalance
)

# Save the trained model
model.save("fraud_detection_cnn.h5")
print("Model saved as fraud_detection_cnn.h5")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

