import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# CONFIGURATION
# =========================
SEQ_LEN = 14
EPOCHS = 20
BATCH_SIZE = 16
SPLIT_DATE = '2016-01-01'

print("TensorFlow Version:", tf.__version__)

# =========================
# STEP 1: Lightweight Data Loading
# =========================
print("Loading data...")

# Load calendar for dates (lightweight)
calendar = pd.read_csv("calendar.csv", usecols=['d', 'date'])

# Load sales train (filter one store to reduce memory immediately)
# Only loading necessary columns if possible, but we need to filter first.
# Just load all and filter store quickly.
sales = pd.read_csv("sales_train_evaluation.csv")
sales = sales[sales['store_id'] == 'CA_1']

# Filter to top 3 items to keep it very lightweight as discussed
top_items = sales['item_id'].value_counts().head(3).index
sales = sales[sales['item_id'].isin(top_items)]

# Melt to long format — only d_* columns are actual sales
print("Melting data...")
d_cols = [c for c in sales.columns if c.startswith('d_')]
sales_long = sales.melt(
    id_vars=['item_id', 'store_id'],
    value_vars=d_cols,
    var_name='d',
    value_name='sales'
)

# Merge only dates
sales_long = sales_long.merge(calendar, on='d', how='left')
sales_long['date'] = pd.to_datetime(sales_long['date'])
sales_long['sales'] = pd.to_numeric(sales_long['sales'], errors='coerce')
sales_long = sales_long.dropna(subset=['sales', 'date'])

# Sort by item and date
sales_long = sales_long.sort_values(['item_id', 'date']).reset_index(drop=True)

# Select ONLY sales column for modeling
data = sales_long[['item_id', 'date', 'sales']].copy()

# =========================
# STEP 2: Preprocessing
# =========================
print("Preprocessing...")

# Normalize sales (MinMax 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
data['sales_scaled'] = scaler.fit_transform(data[['sales']])

# Create Sequences
def create_dataset(group):
    X, y = [], []
    vals = group['sales_scaled'].values
    for i in range(len(vals) - SEQ_LEN):
        X.append(vals[i : i + SEQ_LEN])
        y.append(vals[i + SEQ_LEN])
    return np.array(X), np.array(y)

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []
y_test_actual_list = [] # Store actuals for inverse transform later

for item_id, group in data.groupby('item_id'):
    train_group = group[group['date'] < SPLIT_DATE]
    test_group = group[group['date'] >= SPLIT_DATE]
    
    # Train sequences
    if len(train_group) > SEQ_LEN:
        X_tr, y_tr = create_dataset(train_group)
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
    
    # Test sequences (Careful handling to include lookback from train)
    # We need the last SEQ_LEN points from train to predict the first test point
    if len(test_group) > 0:
        # Get full series for this item to handle the overlap
        full_vals = group['sales_scaled'].values
        split_idx = len(train_group)
        
        # We want to predict for all test dates using sliding window
        # Indices for test predictions start at split_idx
        X_te, y_te = [], []
        
        # Lookback range for test set
        start_idx = split_idx - SEQ_LEN
        
        if start_idx >= 0:
            test_subset = full_vals[start_idx:]
            
            for i in range(len(test_subset) - SEQ_LEN):
                 X_te.append(test_subset[i : i + SEQ_LEN])
                 y_te.append(test_subset[i + SEQ_LEN])
            
            X_test_list.append(np.array(X_te))
            y_test_list.append(np.array(y_te))
            
            # Get actual unscaled sales for evaluation
            # The indices align with test_group[0] to test_group[end]
            # Since we iterate exactly len(test_group) times (conceptually)
            # Actually, len(test_subset) - SEQ_LEN == len(test_group)
            
            actuals = group['sales'].values[split_idx : split_idx + len(y_te)]
            y_test_actual_list.append(actuals)

# Concatenate
X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)
y_test_actual = np.concatenate(y_test_actual_list) 

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], SEQ_LEN, 1)
X_test = X_test.reshape(X_test.shape[0], SEQ_LEN, 1)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# =========================
# STEP 3: Model Building
# =========================
print("Building LSTM...")

model = Sequential([
    LSTM(32, input_shape=(SEQ_LEN, 1)), # Single LSTM layer, 32 units
    Dense(1) # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =========================
# STEP 4: Training
# =========================
print("Training...")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Use 10% of train for validation
    verbose=1
)

# =========================
# STEP 5: Evaluation
# =========================
print("Evaluating...")

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Reference LSTM Training Loss')
plt.legend()
plt.savefig('lstm_training_loss.png')
print("Saved lstm_training_loss.png")

# Predictions
preds_scaled = model.predict(X_test)

# Inverse Transform
preds = scaler.inverse_transform(preds_scaled)
# y_test_actual is already unscaled

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test_actual, preds)
rmse = np.sqrt(mean_squared_error(y_test_actual, preds))

print(f"\nTest MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# Plot Predictions (Subset)
plt.figure(figsize=(12, 5))
plt.plot(y_test_actual[:100], label='Actual Sales')
plt.plot(preds[:100], label='Predicted Sales')
plt.title('LSTM Predictions vs Actual (First 100 Test Points)')
plt.legend()
plt.savefig('lstm_test_predictions.png')
print("Saved lstm_test_predictions.png")

print("Done.")
