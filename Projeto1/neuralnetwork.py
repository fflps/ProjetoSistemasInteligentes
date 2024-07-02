import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
# import matplotlib.pyplot as plt
import time

# Get the dataset from the .data file
dataset = pd.read_csv("D:/00_UFPE/SistemasInteligentes/ProjetoSI/abalone.data")

# Converts the "Sex" nominal variable to discrete values
convert_sex = lambda x: 0 if x == 'M' else 1 if x == 'I' else 2
dataset['Sex'] = dataset['Sex'].apply(convert_sex)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting the data into training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

# Normalizing the data
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Neural Network (with backpropagation) Classifier
classifier = MLPRegressor(hidden_layer_sizes=(15, 15, 15), activation='tanh', alpha=0.15, learning_rate='invscaling', max_iter=1000)

start = time.time()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
end = time.time()
print(f"Time: {(end-start)*1000:.3f} ms")
print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("MAE: ", mean_absolute_error(Y_test, Y_pred))
print("MAPE: ", mean_absolute_percentage_error(Y_test, Y_pred))
print("R2: ", r2_score(Y_test, Y_pred))
