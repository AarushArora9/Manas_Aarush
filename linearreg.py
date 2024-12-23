import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:/Users/aarus/Downloads/Seniors.csv')

# Define the feature columns and target column
feature_columns = ['Temperature (°C)', 'Pressure (kPa)', 'Temperature x Pressure', 'Material Fusion Metric', 'Material Transformation Metric']
target_column = 'Quality Rating'

# Function to remove outliers
def remove_outliers(df, feature_columns):
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Remove outliers from the dataset
df = remove_outliers(df, feature_columns)

# Select features and target
X = df[feature_columns].to_numpy()
y = df[target_column].to_numpy()
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Check for NaNs or infinite values in the dataset
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Warning: NaNs or infinite values found in the features!")
    X = np.nan_to_num(X)  # Replace NaNs and infinities with zero

if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    print("Warning: NaNs or infinite values found in the target!")
    y = np.nan_to_num(y)  # Replace NaNs and infinities with zero

# Feature scaling: Standardization (mean 0, std 1)
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

X_scaled, X_mean, X_std = standardize(X)

# Add intercept term (bias term) to X (x_0 = 1)
m = X_scaled.shape[0]
X_scaled = np.concatenate([np.ones((m, 1)), X_scaled], axis=1)

# Initialize parameters
def initialize_theta(n):
    return np.zeros(n)

# Compute cost function with regularization
def compute_cost(X, y, Theta, lambda_):
    m = len(y)
    h = X.dot(Theta)
    error = h - y
    cost = (1 / (2 * m)) * np.sum(np.square(error))  # Mean squared error
    regularization = (lambda_ / (2 * m)) * np.sum(np.square(Theta[1:]))  # Regularization term (L2)
    return cost + regularization

# Mini-Batch Gradient Descent with L2 Regularization (Ridge Regression)
def mini_batch_gradient_descent(X, y, Theta, alpha, epochs, lambda_, batch_size):
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, m, batch_size):
            # Get the mini-batch
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Compute the prediction
            h = X_batch.dot(Theta)
            error = h - y_batch

            # Compute gradient
            gradient = (1 / batch_size) * X_batch.T.dot(error)

            # Regularization term: Apply to all but the intercept
            regularization_term = (lambda_ / m) * np.concatenate([[0], Theta[1:]])

            # Update the parameters
            Theta -= alpha * (gradient + regularization_term)

        # Compute the cost after each epoch
        cost = compute_cost(X, y, Theta, lambda_)
        cost_history.append(cost)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} | Cost: {cost:.4f}")

    return Theta, cost_history

# Hyperparameters
alpha = 0.01  # Learning rate
epochs = 100000  # Number of iterations
lambda_ = 1  # Regularization strength (lambda)
batch_size = 64  # Mini-batch size

# Initialize Theta
Theta = initialize_theta(X_scaled.shape[1])

# Run Mini-Batch Gradient Descent
Theta_opt, cost_history = mini_batch_gradient_descent(X_scaled, y, Theta, alpha, epochs, lambda_, batch_size)

# Make predictions
y_pred = X_scaled.dot(Theta_opt)

# Cap the predictions at 100
y_pred = np.clip(y_pred, None, 100)

# Calculate performance metrics
mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

# Print the metrics
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot cost history to observe convergence
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), cost_history, label='Cost function')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.legend()
plt.show()

# Plot Actual vs Predicted values with Best Fit Line (using the average slope)

slopes = []
for i in range(1, len(y)):
    x1, y1 = y[i-1], y_pred[i-1]  
    x2, y2 = y[i], y_pred[i]      
    if x2 != x1:  # Avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

# Calculate the average slope
avg_slope = np.mean(slopes)

# Compute the intercept using the average slope and the first point
intercept = y_pred[0] - avg_slope * y[0]

# Generate points for the best fit line
line_x = np.linspace(min(y), max(y), 100)
line_y = avg_slope * line_x + intercept  

# Plot the Actual vs Predicted values with the Best Fit Line
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='b', label="Actual vs Predicted", alpha=0.5)
plt.plot(line_x, line_y, color='r', label="Best Fit Line (Average Slope)")
plt.xlabel("Actual Quality Rating")
plt.ylabel("Predicted Quality Rating")
plt.title("Actual vs Predicted Quality Ratings with Best Fit Line (Average Slope)")
plt.legend()
plt.show()
new_data_df = pd.read_csv('C:/Users/aarus/Downloads/Juniors.csv')
X_new = new_data_df[feature_columns].to_numpy()
X_new = (X_new - X_min) / (X_max - X_min)
X_new_with_intercept = np.concatenate((np.ones((X_new.shape[0], 1)), X_new), axis=1)
predicted_rating = np.dot(X_new_with_intercept, Theta)
predicted_rating = np.clip(predicted_rating, a_min=0, a_max=100)
new_data_df['Predicted_rating'] = predicted_rating
new_data_df.to_csv('C:/Users/aarus/Downloads/rating.csv', index=False)
print(new_data_df)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    return mae, mse, r2


def test_epochs(X, y, alpha, epoch_range):
    avg_losses = [] 
    for power in epoch_range:
        epochs = 10**power
        Theta_opt, cost = mini_batch_gradient_descent(X, y, initialize_theta(X.shape[1]), alpha, epochs, lambda_, batch_size)
        avg_losses.append(np.mean(cost))  
    return avg_losses

# Test the model for different epochs
epoch_range = np.arange(0, 5)  # Epoch range from 10^0 to 10^6
alpha = 0.001  # Learning rate
avg_losses = test_epochs(X_scaled, y, alpha, epoch_range)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(10**epoch_range, avg_losses, marker='o', color='b')
plt.xscale('log') 
plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss (Cost)')
plt.title('Average Loss vs Epochs (Logarithmic Scale)')
plt.grid(True)
plt.show()

# Function to test different learning rates
def test_alphas(X, y, alpha_range, epochs=1000):
    avg_losses = [] 
    maes = [] 
    mses = [] 
    r2s = []  

    for alpha in alpha_range:
        Theta_opt, cost = mini_batch_gradient_descent(X, y, initialize_theta(X.shape[1]), alpha, epochs, lambda_, batch_size)
        #X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_pred = np.dot(X, Theta_opt)
        mae, mse, r2 = calculate_metrics(y, y_pred)
        avg_losses.append(np.mean(cost)) 
        maes.append(mae)
        mses.append(mse)
        r2s.append(r2)

    return avg_losses, maes, mses, r2s

# Test the model for different learning rates (logarithmic scale)
alpha_range = np.logspace(-3, -1, 20)  # Alpha range from 0.001 to 0.1
avg_losses, maes, mses, r2s = test_alphas(X_scaled, y, alpha_range)

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Zoomed-in view of MAE, MSE, R2
zoom_idx = 0
axes[1].plot(alpha_range[zoom_idx:], maes[zoom_idx:], marker='o', color='g', label='Mean Absolute Error (MAE)')
axes[1].plot(alpha_range[zoom_idx:], mses[zoom_idx:], marker='o', color='r', label='Mean Squared Error (MSE)')
axes[1].plot(alpha_range[zoom_idx:], r2s[zoom_idx:], marker='o', color='purple', label='R-squared')
axes[1].set_xscale('log')
axes[1].set_xlabel('Learning Rate (Alpha) (Log Scale)')
axes[1].set_ylabel('Metric Value')
axes[1].set_title('MAE, MSE, and R-squared (Zoomed-in view)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()