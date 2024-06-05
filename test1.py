import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import time

# Get the dataset from the .data file
dataset = pd.read_csv("D:/00_UFPE/SistemasInteligentes/ProjetoSI/abalone.data")

# Converts the "Sex" nominal variable to discrete values
convert_sex = lambda x: 0 if x == 'M' else 1 if x == 'I' else 2
dataset['Sex'] = dataset['Sex'].apply(convert_sex)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

num_samples, num_features = np.shape(X_train)

# Normalização dos dados
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Comparação entre os Classificadores
classifiers = {
    "Decision Tree": tree.DecisionTreeClassifier(random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": svm.SVC(kernel='linear'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
}

results = []

for name, classifier in classifiers.items():
    start = time.time()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    end = time.time()
    accuracy = accuracy_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    results.append((name, accuracy, mae, end - start))

# Display the results for initial comparison
for result in results:
    print(f"{result[0]}: Accuracy: {result[1]:.4f}, MAE: {result[2]:.4f}, Time: {result[3]*1000:.3f} ms")

# Otimização dos Hiperparâmetros da Rede Neural
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [500, 1000, 1500]
}

mlp = MLPClassifier(random_state=0)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
start = time.time()
grid_search.fit(X_train, Y_train)
end = time.time()

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print("\nBest Hyperparameters for Neural Network:", best_params)
print("Neural Network with Optimized Hyperparameters: Accuracy: {:.4f}, MAE: {:.4f}, Time: {:.3f} ms".format(accuracy, mae, (end-start)*1000))
