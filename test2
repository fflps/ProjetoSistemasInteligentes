import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
import time

# Get the dataset from the .data file
dataset = pd.read_csv("D:/00_UFPE/SistemasInteligentes/ProjetoSI/abalone.data")

# Converts the "Sex" nominal variable to discrete values
convert_sex = lambda x: 0 if x == 'M' else 1 if x == 'I' else 2
dataset['Sex'] = dataset['Sex'].apply(convert_sex)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Comparation among the Classifiers
classifiers = {
    "Decision Tree": tree.DecisionTreeClassifier(random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": svm.SVC(kernel='linear')
}

results = {}

for name, classifier in classifiers.items():
    start = time.time()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    end = time.time()
    results[name] = {
        "time": (end - start) * 1000,
        "accuracy": accuracy_score(Y_test, Y_pred),
        "mae": mean_absolute_error(Y_test, Y_pred)
    }
    print(name, f"Time: {results[name]['time']:.3f} ms")
    print(name, "Accuracy: ", results[name]['accuracy']) 
    print(name, "MAE: ", results[name]['mae'])

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train, Y_train)
Y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest Accuracy: ", accuracy_score(Y_test, Y_pred_rf))
print("Random Forest MAE: ", mean_absolute_error(Y_test, Y_pred_rf))

# Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)
gb_classifier.fit(X_train, Y_train)
Y_pred_gb = gb_classifier.predict(X_test)
print("Gradient Boosting Accuracy: ", accuracy_score(Y_test, Y_pred_gb))
print("Gradient Boosting MAE: ", mean_absolute_error(Y_test, Y_pred_gb))

# Redução de dimensionalidade para KNN
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Grid Search para KNN
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_train_pca, Y_train)
Y_pred_knn_grid = knn_grid.best_estimator_.predict(X_test_pca)

print("Best KNN Configuration:", knn_grid.best_params_)
print("Best KNN Accuracy: ", accuracy_score(Y_test, Y_pred_knn_grid))
print("Best KNN MAE: ", mean_absolute_error(Y_test, Y_pred_knn_grid))

# Grid Search para SVM
svm_params = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm.SVC(), svm_params, cv=5)
svm_grid.fit(X_train, Y_train)
Y_pred_svm_grid = svm_grid.best_estimator_.predict(X_test)

print("Best SVM Configuration:", svm_grid.best_params_)
print("Best SVM Accuracy: ", accuracy_score(Y_test, Y_pred_svm_grid))
print("Best SVM MAE: ", mean_absolute_error(Y_test, Y_pred_svm_grid))
