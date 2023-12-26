#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def lasso_feature_selection(TF_matrix, Lnc_matrix, alpha=0.01, top_features=10):
    # Read TF matrix and lnc matrix from files
    df_tf = pd.read_table(TF_matrix, sep='\t', index_col=0)
    df_lnc = pd.read_table(Lnc_matrix, sep='\t', index_col=0)

    # Transpose matrices to have genes as rows and time points as columns
    df_tf = df_tf.T
    df_lnc = df_lnc.T

    # Assuming df_tf is your TF matrix and df_lnc is your lnc matrix
    X = df_lnc.values
    y = df_tf.values

    # Check if the number of rows in X and y are the same
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X and y must be the same.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LASSO Regression
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = lasso.predict(X_test_scaled)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)

    # Identify Important Features (genes)
    selected_features_indices = np.where(np.abs(lasso.coef_) != 0)[0]
    
    # Check if top_features is greater than the available features
    if top_features > len(selected_features_indices):
        top_features = len(selected_features_indices)
        print(f"Reducing top_features to {top_features} as it exceeds the available features.")

    # Sort features based on coefficients and select the top features
    top_features_indices = selected_features_indices[np.argsort(np.abs(lasso.coef_[selected_features_indices]))[::-1]][:top_features]
    selected_feature_names = np.array(df_lnc.columns)[top_features_indices]
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig('Performance.png', format='png')

    # Remove duplicates from the selected feature names
    selected_feature_names = np.unique(selected_feature_names)

    # Print or use the selected feature names
    print( f"{len(selected_feature_names)} lncRNA/TFs were selected using LASSO: ",", ".join(selected_feature_names))

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python LASSO_feature_selection.py TF_matrix Lnc_matrix")
        sys.exit(1)

    # Extract file paths from command-line arguments
    TF_matrix = sys.argv[1]
    Lnc_matrix = sys.argv[2]

    # Run the lasso_feature_selection function with the provided file paths
    lasso_feature_selection(TF_matrix, Lnc_matrix, top_features=10)
