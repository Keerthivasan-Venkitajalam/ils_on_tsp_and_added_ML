import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pathlib
import glob

# Load results from the results directory
def load_results(results_dir):
    results_files = glob.glob(f"{results_dir}/results_*.xlsx")
    data = []

    for file in results_files:
        df = pd.read_csv(file, header=None)
        data.append(df)

    # Combine all dataframes into a single dataframe
    return pd.concat(data, ignore_index=True)

# Preprocess data for machine learning
def preprocess_data(data):
    # Assuming the first column is the initial city and the second is the route cost
    data.columns = ['Initial_City', 'Route_Cost', 'Processing_Time']
    # Convert to numeric types
    data['Route_Cost'] = pd.to_numeric(data['Route_Cost'], errors='coerce')
    data['Processing_Time'] = pd.to_numeric(data['Processing_Time'], errors='coerce')
    # Drop rows with NaN values
    return data.dropna()

# Train the machine learning model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Plot results
def plot_results(data, model):
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of actual vs predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(data['Route_Cost'], model.predict(data[['Initial_City', 'Processing_Time']]), alpha=0.5)
    plt.xlabel('Actual Route Cost')
    plt.ylabel('Predicted Route Cost')
    plt.title('Actual vs Predicted Route Cost')
    plt.plot([data['Route_Cost'].min(), data['Route_Cost'].max()], 
             [data['Route_Cost'].min(), data['Route_Cost'].max()], 
             color='red', linestyle='--')
    
    # Feature importance plot
    plt.subplot(1, 2, 2)
    importances = model.feature_importances_
    features = ['Initial_City', 'Processing_Time']
    sns.barplot(x=importances, y=features)
    plt.title('Feature Importances')

    plt.tight_layout()
    plt.savefig('results/ML_model_results.png')
    plt.show()

def main():
    results_dir = pathlib.Path(__file__).parent.resolve() / "results"
    results_data = load_results(results_dir)
    processed_data = preprocess_data(results_data)

    # Define features and target
    X = processed_data[['Initial_City', 'Processing_Time']]
    y = processed_data['Route_Cost']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')

    # Plot the results
    plot_results(processed_data, model)

if __name__ == "__main__":
    main()
