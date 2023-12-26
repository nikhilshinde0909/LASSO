#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

## Define help
def print_help():
    print("Usage: python LASSO_feature_selection.py <TF_matrix> <Lnc_matrix> <number of features>")
    
if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
    print_help()
    sys.exit(0)
    
if len(sys.argv) != 4:
    print("Invalid number of arguments!")
    print_help()
    sys.exit(1)

TF_matrix = sys.argv[1]
Lnc_matrix = sys.argv[2]
num_features= int(sys.argv[3])

def lasso_feature_selection(TF_matrix, Lnc_matrix, top_features=num_features, alpha=0.01):
    df_tf = pd.read_table(TF_matrix, sep='\t', index_col=0)
    df_lnc = pd.read_table(Lnc_matrix, sep='\t', index_col=0)
    df_tf = df_tf.T
    df_lnc = df_lnc.T
    X = df_lnc.values
    y = df_tf.values
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X and y must be the same.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    y_pred = lasso.predict(X_test_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)

    selected_features_indices = np.where(np.abs(lasso.coef_) != 0)[0]

    if top_features > len(selected_features_indices):
        top_features = len(selected_features_indices)
        print(f"Reducing top_features to {top_features} as it exceeds the available features.")

    top_features_indices = selected_features_indices[np.argsort(np.abs(lasso.coef_[selected_features_indices]))[::-1]][:top_features]
    selected_feature_names = np.array(df_lnc.columns)[top_features_indices]
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig('Feature-performance.png', format='png')

    selected_feature_names = np.unique(selected_feature_names)

    print(f"{len(selected_feature_names)} lncRNA/TFs were selected using LASSO: {', '.join(selected_feature_names)}")
     
lasso_feature_selection(TF_matrix, Lnc_matrix, top_features=num_features)
