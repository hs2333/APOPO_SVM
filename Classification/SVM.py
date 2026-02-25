# 6-classification_svm.py
# Capstone: 6-state motion classification with SVM (RBF)
# - Uses an 11D feature set (acc mag, gyro stats, gravity angles)
# - Standardizes features (required for SVM)
# - Evaluates SVM on 11D features
# - Saves 2D decision-region plots in PCA space (so plots reflect more of the 11D structure)
#   (No plt.show(); plots are saved as PNG files)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# -----------------------------
# Paths
# -----------------------------
X_TRAIN_PATH = "X_train.txt"
Y_TRAIN_PATH = "y_train.txt"
FEATURES_PATH = "features.txt"


# -----------------------------
# Utilities
# -----------------------------
def load_feature_names(features_path: str) -> list[str]:
    names = []
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                names.append(parts[1])
    return names


# -----------------------------
# Load feature names and data
# -----------------------------
feature_names = load_feature_names(FEATURES_PATH)

X_full = np.loadtxt(X_TRAIN_PATH)
y = np.loadtxt(Y_TRAIN_PATH, dtype=int)  # 6-class labels (1..6)

label_names = {
    1: "walking",
    2: "upstairs",
    3: "downstairs",
    4: "sitting",
    5: "standing",
    6: "laying"
}
labels = [1, 2, 3, 4, 5, 6]
target_names = [label_names[i] for i in labels]


# -----------------------------
# Feature set (11D)
# -----------------------------
FEATURE_SET = [
    "tBodyAccMag-mean()", "tBodyAccMag-std()",
    "tBodyGyro-mean()-X", "tBodyGyro-mean()-Y", "tBodyGyro-mean()-Z",
    "tBodyGyro-std()-X",  "tBodyGyro-std()-Y",  "tBodyGyro-std()-Z",
    "angle(X,gravityMean)", "angle(Y,gravityMean)", "angle(Z,gravityMean)"
]

feat_idxs = []
missing = []
for name in FEATURE_SET:
    if name in feature_names:
        feat_idxs.append(feature_names.index(name))
    else:
        missing.append(name)

if missing:
    raise ValueError(f"These features were not found in features.txt: {missing}")

print("Using features:")
for name, idx0 in zip(FEATURE_SET, feat_idxs):
    print(f"  {name:<30} -> column {idx0+1} (1-based)")

X = X_full[:, feat_idxs].astype(float)


# -----------------------------
# Split (use indices so we can reuse for PCA plotting cleanly)
# -----------------------------
idx_all = np.arange(len(X))
idx_train, idx_test = train_test_split(
    idx_all,
    test_size=0.10,
    random_state=0,
    stratify=y
)

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

N_train = len(X_train)
print(f"\nN_train = {N_train}")


# -----------------------------
# Standardize (required for SVM)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Train + Evaluate SVM (RBF)
# -----------------------------
C = 10
gamma = "scale"
svm = SVC(kernel="rbf", C=C, gamma=gamma)

svm.fit(X_train, y_train)
pred = svm.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\nSVM (RBF) Holdout (10%) results:")
print(f"  C={C}, gamma={gamma}")
print(f"  Accuracy: {acc:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred, labels=labels))

print("\nClassification report:")
print(classification_report(y_test, pred, labels=labels, target_names=target_names))


# -----------------------------
# PCA (2D) for visualization + saved decision-region plots
# -----------------------------
pca = PCA(n_components=2, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
print("PCA total explained variance (2D):", float(np.sum(pca.explained_variance_ratio_)))


def save_decision_region_plot_2d(model2d, X2_train, X2_test, y_train, y_test, filename: str, title: str):
    pad = 0.35
    x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
    y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 800),
        np.linspace(y_min, y_max, 800)
    )
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    Z = model2d.predict(grid2).reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    levels = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    plt.contourf(xx, yy, Z, alpha=0.25, levels=levels)

    for c in labels:
        plt.scatter(
            X2_train[y_train == c, 0], X2_train[y_train == c, 1],
            s=14, label=f"{c}: {label_names[c]} (train)"
        )
    for c in labels:
        plt.scatter(
            X2_test[y_test == c, 0], X2_test[y_test == c, 1],
            s=35, marker="x", label=f"{c}: {label_names[c]} (test)"
        )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# IMPORTANT:
# The SVM you trained above is in 11D. For a valid 2D decision-region plot, train a NEW SVM in PCA space.
svm_pca = SVC(kernel="rbf", C=C, gamma=gamma)
svm_pca.fit(X_train_pca, y_train)

save_decision_region_plot_2d(
    svm_pca,
    X_train_pca,
    X_test_pca,
    y_train,
    y_test,
    filename="svm_rbf_pca.png",
    title=f"SVM (RBF) decision regions in PCA space (C={C}, gamma={gamma})"
)

print("\nSaved plot:")
print("  svm_rbf_pca.png")
