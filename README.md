# creditcard-fraud-detector
creditcard-fraud-detector

#Running command
-----------------------------------------------------------------------------------------
python3 -m venv fraud_env
source fraud_env/bin/activate
pip install tensorflow numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
