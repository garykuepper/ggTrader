import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# ---------------------------
# KNN Classification Example
# ---------------------------
X_class, y_class = make_classification(
    n_samples=200, n_features=2,
    n_redundant=0, n_clusters_per_class=1,
    n_classes=2, random_state=42
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42
)

scaler_c = StandardScaler()
Xc_train = scaler_c.fit_transform(Xc_train)
Xc_test = scaler_c.transform(Xc_test)

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(Xc_train, yc_train)

# Meshgrid for decision boundary
xx, yy = np.meshgrid(
    np.linspace(Xc_train[:, 0].min()-1, Xc_train[:, 0].max()+1, 200),
    np.linspace(Xc_train[:, 1].min()-1, Xc_train[:, 1].max()+1, 200)
)
Z = knn_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ---------------------------
# KNN Regression Example
# ---------------------------
X_reg, y_reg = make_regression(
    n_samples=200, n_features=1, noise=15, random_state=42
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

scaler_r = StandardScaler()
Xr_train = scaler_r.fit_transform(Xr_train)
Xr_test = scaler_r.transform(Xr_test)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(Xr_train, yr_train)

# Predict on grid
Xr_line = np.linspace(Xr_train.min(), Xr_train.max(), 200).reshape(-1, 1)
y_pred_line = knn_reg.predict(Xr_line)

# ---------------------------
# Plotting
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Classification plot
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
axes[0].scatter(Xc_train[:, 0], Xc_train[:, 1], c=yc_train,
                cmap=plt.cm.coolwarm, edgecolors='k')
axes[0].set_title("KNN Classification (k=5)")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# Regression plot
axes[1].scatter(Xr_train, yr_train, c="blue", label="Train data")
axes[1].plot(Xr_line, y_pred_line, c="red", label="KNN Regression fit")
axes[1].set_title("KNN Regression (k=5)")
axes[1].set_xlabel("Feature")
axes[1].set_ylabel("Target")
axes[1].legend()

plt.tight_layout()
plt.show()
