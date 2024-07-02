import numpy as np
import pandas as pd
from sklearn import tree, svm, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

# Normalização dos dados
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)


# Comparation among the Classifiers
classifiers = {
    "Decision Tree": tree.DecisionTreeClassifier(random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": svm.SVC(kernel='linear'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
}

for name, classifier in classifiers.items():
    start = time.time()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    end = time.time()
    print(name, f"Time: {(end-start)*1000:.3f} ms")
    print(name, "Accuracy: ", accuracy_score(Y_test, Y_pred)) 
    print(name, "MAE: ", mean_absolute_error(Y_test, Y_pred))
