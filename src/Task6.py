import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Loading and cleaning dataset
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])

X = df[["SepalLengthCm", "SepalWidthCm"]].values
y = df["Species"]

#Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

k_values = range(1, 21)
acc_list = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))

best_k = k_values[int(np.argmax(acc_list))]
print("Best K:", best_k)

best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))

Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])

cats = pd.Categorical(y).categories
label_to_code = {label: i for i, label in enumerate(cats)}
Z_codes = np.array([label_to_code[label] for label in Z]).reshape(xx.shape)
y_codes = y.map(label_to_code).values

plt.contourf(xx, yy, Z_codes, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y_codes, edgecolors="k", s=35)
plt.xlabel("Sepal Length in Cm (standardized)")
plt.ylabel("Sepal Width in Cm (standardized)")
plt.title(f"KNN Decision Boundary (K={best_k})")
plt.show()