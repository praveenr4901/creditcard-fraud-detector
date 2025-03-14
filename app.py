import streamlit as st
import pandas as pd
import os
import re
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from login import check_authentication, get_user_role, logout

# Initialize session state for model tracking
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Authenticate user and get role
check_authentication()
user_role = get_user_role()

st.sidebar.title(f"Welcome, {st.session_state.get('username', 'User')}")
logout()

# File paths
TRANSACTION_FILE = "transactions.csv"
FRAUD_USERS_FILE = "fraud_users.csv"
MODEL_FILE = "fraud_detection_cnn.h5"

# Define preprocessing function first
def preprocess_data_for_model(df):
    df = df.copy()

    # Ensure transaction_amount is numeric
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(0)
    
    # Handle categorical features - ensure they're converted to numeric values
    if 'country' in df.columns:
        df['country'] = pd.Categorical(df['country']).codes
    else:
        df['country'] = 0
        
    if 'payment_method' in df.columns:
        df['payment_method'] = pd.Categorical(df['payment_method']).codes
    else:
        df['payment_method'] = 0

    # Normalize the transaction amount (ensure all values are numeric)
    min_val = float(df['transaction_amount'].min())
    max_val = float(df['transaction_amount'].max())
    
    # Avoid division by zero
    if max_val > min_val:
        df['transaction_amount'] = (df['transaction_amount'] - min_val) / (max_val - min_val)
    else:
        df['transaction_amount'] = 0

    # Select feature columns
    feature_columns = ['transaction_amount', 'country', 'payment_method']
    
    # Ensure all selected columns contain numeric data
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    data = df[feature_columns].values

    # Create a fixed-size feature vector (384 features)
    expected_features = 384
    if data.shape[1] < expected_features:
        # Pad with zeros if there are fewer features
        padding = np.zeros((data.shape[0], expected_features - data.shape[1]))
        data = np.hstack((data, padding))
    elif data.shape[1] > expected_features:
        # Truncate if there are more features
        data = data[:, :expected_features]

    # Reshape the data for the model (for Conv1D layers)
    data = data.reshape((data.shape[0], data.shape[1], 1))

    return data

# Function to create and train a new model
def create_and_train_model():
    if not st.session_state.get('model_being_created', False):
        st.session_state.model_being_created = True
        st.info("Generating new fraud detection model...")
        
        # Check if transaction data exists for training
        if not os.path.exists(TRANSACTION_FILE):
            st.warning("No transaction data available for model training. Using default model.")
            # Create a simple default model
            model = Sequential()
            model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(384, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Save the default model
            model.save(MODEL_FILE)
            st.session_state.model_loaded = True
            st.session_state.model_being_created = False
            return model
        
        try:
            # Load transaction data
            df = pd.read_csv(TRANSACTION_FILE)
            if df.empty:
                raise ValueError("Transaction file is empty")
                
            # Ensure transaction_amount is numeric
            df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(0)
            
            # Preprocess data for training
            processed_data = preprocess_data_for_model(df)
            
            # Create synthetic labels for training
            # Use a simple heuristic: transactions over a certain amount are more likely to be fraud
            high_amount_threshold = df['transaction_amount'].quantile(0.8)
            
            # Label as potential fraud if amount is high
            fraud_labels = (df['transaction_amount'] > high_amount_threshold).astype(int).values
            
            # Create and train the model
            model = Sequential()
            model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(processed_data.shape[1:]), 
                        name='conv1d_input'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(64, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Simple training
            model.fit(processed_data, fraud_labels, epochs=5, batch_size=32, verbose=0)
            
            # Save the trained model
            model.save(MODEL_FILE)
            st.success("New fraud detection model generated successfully!")
            
            st.session_state.model_loaded = True
            st.session_state.model_being_created = False
            return model
            
        except Exception as e:
            st.warning(f"Error processing transaction data: {e}. Using default model.")
            # Create a simple default model as fallback
            model = Sequential()
            model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(384, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Save the default model
            model.save(MODEL_FILE)
            st.session_state.model_loaded = True
            st.session_state.model_being_created = False
            return model

# Load or create the model only once
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE)
            st.session_state.model_loaded = True
            return model
        except Exception as e:
            if not st.session_state.model_loaded:
                st.error(f"Error loading existing model: {e}. Creating a new model...")
                return create_and_train_model()
    else:
        if not st.session_state.model_loaded:
            st.info("No existing model found. Creating a new model...")
            return create_and_train_model()

# Load the model only once
model = get_model()

def is_valid_credit_card(cc_number):
    return bool(re.fullmatch(r"\d{16}", cc_number))

def is_valid_phone(phone_number):
    return bool(re.fullmatch(r"\d{10,15}", phone_number))

def is_valid_email(email):
    return bool(re.fullmatch(r"[^@]+@[^@]+\.[a-zA-Z]{2,}", email))

def is_fraud_user(cc_number, phone_number, email):
    fraud_db = pd.read_csv(FRAUD_USERS_FILE) if os.path.exists(FRAUD_USERS_FILE) else pd.DataFrame()
    if fraud_db.empty:
        return False
    return (
        cc_number in fraud_db.get("cc_number", []) or
        phone_number in fraud_db.get("phone_number", []) or
        email in fraud_db.get("email", [])
    )

def store_transaction(data):
    save_mode = 'a' if os.path.exists(TRANSACTION_FILE) else 'w'
    header = not os.path.exists(TRANSACTION_FILE)
    data.to_csv(TRANSACTION_FILE, mode=save_mode, header=header, index=False)

def predict_fraud():
    if not os.path.exists(TRANSACTION_FILE) or model is None:
        return None

    try:
        df = pd.read_csv(TRANSACTION_FILE)
        if df.empty:
            st.info("No transactions recorded yet.")
            return None
    
        # Ensure all numeric fields are properly converted
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(0)
        
        processed_data = preprocess_data_for_model(df)
        
        # Get expected input shape - use hardcoded value since we know our preprocessing creates data of shape (n, 384, 1)
        expected_features = 384
        if processed_data.shape[1] != expected_features:
            st.error(f"🚨 Data shape mismatch! Model expects {expected_features} features but got {processed_data.shape[1]}")
            return None
            
        predictions = model.predict(processed_data)
        df['fraud_probability'] = predictions.flatten()
        df['fraud_status'] = df['fraud_probability'].apply(lambda x: "Fraud" if x > 0.5 else "Not Fraud")
        
        return df
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Set up a clean container for user interface to remove old notifications
if user_role == "user":
    main_container = st.container()
    with main_container:
        st.title("Enter Transaction Details")
        cc_number = st.text_input("Credit Card Number (16 Digits)")
        phone_number = st.text_input("Phone Number (10-15 Digits)")
        email = st.text_input("Email Address")
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
        billing_address = st.text_input("Billing Address")
        shipping_address = st.text_input("Shipping Address")
        country = st.text_input("Country")
        payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal"])

        if st.button("Submit Transaction"):
            errors = []
            if not is_valid_credit_card(cc_number):
                errors.append("Invalid Credit Card Number! Must be 16 digits.")
            if not is_valid_phone(phone_number):
                errors.append("Invalid Phone Number! Must be 10-15 digits.")
            if not is_valid_email(email):
                errors.append("Invalid Email Address! Please enter a valid email.")
            if is_fraud_user(cc_number, phone_number, email):
                errors.append("🚨 Fraud detected! Transaction not allowed.")
            if billing_address != shipping_address:
                errors.append("🚨 Billing address does not match shipping address.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                transaction_data = pd.DataFrame([{
                    "cc_number": cc_number,
                    "phone_number": phone_number,
                    "email": email,
                    "transaction_amount": float(transaction_amount),  # Ensure numeric
                    "billing_address": billing_address,
                    "shipping_address": shipping_address,
                    "country": country,
                    "payment_method": payment_method
                }])
                store_transaction(transaction_data)
                st.success("✅ Transaction saved successfully!")

elif user_role == "admin":
    main_container = st.container()
    with main_container:
        st.title("Admin Panel - Transaction Records")
        
        # Add a refresh button for admins to regenerate the model on demand
        if st.button("Regenerate Fraud Detection Model"):
            try:
                # Reset the model loaded state
                st.session_state.model_loaded = False
                st.session_state.model_being_created = False
                # Force model recreation
                model = create_and_train_model()
                st.success("Model regenerated successfully!")
            except Exception as e:
                st.error(f"Error creating model: {e}")
        
        # Show fraud prediction results
        st.write("### Transaction Fraud Detection Results")
        fraud_df = predict_fraud()
        
        if fraud_df is not None and not fraud_df.empty:
            # Add filters for admin to view data
            st.write("Filter transactions:")
            fraud_filter = st.selectbox("Show", ["All Transactions", "Fraud Only", "Non-Fraud Only"])
            
            filtered_df = fraud_df
            if fraud_filter == "Fraud Only":
                filtered_df = fraud_df[fraud_df['fraud_status'] == "Fraud"]
            elif fraud_filter == "Non-Fraud Only":
                filtered_df = fraud_df[fraud_df['fraud_status'] == "Not Fraud"]
                
            # Display fraud stats
            fraud_count = len(fraud_df[fraud_df['fraud_status'] == "Fraud"])
            total_count = len(fraud_df)
            fraud_percentage = (fraud_count / total_count * 100) if total_count > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total_count)
            col2.metric("Fraud Transactions", fraud_count)
            col3.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
            
            # Show the data
            st.dataframe(filtered_df[['cc_number', 'phone_number', 'email', 'transaction_amount', 
                                'country', 'payment_method', 'fraud_status', 'fraud_probability']])
            
            # Add ability to mark users as fraud
            st.write("### Mark User as Fraud")
            if not filtered_df.empty:
                selected_user = st.selectbox("Select user to mark as fraud", 
                                        [f"{row['email']} (Card: {row['cc_number']})" 
                                        for _, row in filtered_df.iterrows()])
                
                if st.button("Add to Fraud Database"):
                    email = selected_user.split(" (Card:")[0]
                    row_data = filtered_df[filtered_df['email'] == email].iloc[0]
                    
                    fraud_data = pd.DataFrame([{
                        "cc_number": row_data['cc_number'],
                        "phone_number": row_data['phone_number'],
                        "email": row_data['email']
                    }])
                    
                    save_mode = 'a' if os.path.exists(FRAUD_USERS_FILE) else 'w'
                    header = not os.path.exists(FRAUD_USERS_FILE)
                    fraud_data.to_csv(FRAUD_USERS_FILE, mode=save_mode, header=header, index=False)
                    
                    st.success(f"User {email} added to fraud database!")
            else:
                st.info("No transactions match the current filter.")
        else:
            st.info("No transactions recorded yet.")
