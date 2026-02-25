# K-nn for Dynamic vs Static classification
# Features:
#   - tBodyAccMag-mean()  (feature #201 in features.txt)
#   - tBodyAccMag-std()   (feature #202 in features.txt)
# Labels:
#   Dynamic (1): {1,2,3} = walking, upstairs, downstairs
#   Static  (0): {4,5,6} = sitting, standing, laying

# Advanced version for capstone: 6 states
# 6-classification.py
# Capstone: 6-state motion classification with KNN
# - Train/evaluate on a richer feature set (11D)
# - Standardize features (critical for KNN)
# - Use weights="distance"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler    # Standardize the data
from sklearn.decomposition import PCA   # For better 2d plot
from sklearn.svm import SVC     # Using SVM instead of K-nn

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
y_activity = np.loadtxt(Y_TRAIN_PATH, dtype=int)

# 6-class labels (1..6)
y = y_activity.copy()

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
# Feature set (11D) for training
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

# Keep a 2D slice ONLY for plotting (first two features)
# This is still in the ORIGINAL (unscaled) space right now.
X_plot2d = X[:, [0, 1]].copy()
PLOT_X_LABEL = FEATURE_SET[0]
PLOT_Y_LABEL = FEATURE_SET[1]


# -----------------------------
# Train/test split USING INDICES (so we can split X and X_plot2d consistently)
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

X2_train, X2_test = X_plot2d[idx_train], X_plot2d[idx_test]


# -----------------------------
# Standardize (separate scalers for 11D vs 2D)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

# ---- PCA for 2D visualization (fit on TRAIN only) ----
pca = PCA(n_components=2, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
print("PCA total explained variance (2D):", pca.explained_variance_ratio_.sum())


# -----------------------------
# Choose k values
# -----------------------------
N = len(X_train)
k1 = 1
kN = min(N, 200)   # keep your "large k" demo but avoid absurdly huge k
k_valid = max(3, int(np.sqrt(N)))  # simple heuristic

print(f"\nN_train = {N}")
print(f"k choices: k=1, k=N={kN}, k_valid={k_valid}")


# -----------------------------
# Fit + evaluate on the FULL 11D feature set
# -----------------------------
def fit_and_eval_knn(Xtr, ytr, Xte, yte, k: int):
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    #model_name = "knn"
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    return model, pred, acc

m1, p1, acc1 = fit_and_eval_knn(X_train, y_train, X_test, y_test, k1)
mN, pN, accN = fit_and_eval_knn(X_train, y_train, X_test, y_test, kN)
mv, pv, accv = fit_and_eval_knn(X_train, y_train, X_test, y_test, k_valid)

print("\nHoldout (10%) accuracy:")
print(f"  k=1       : {acc1:.4f}")
print(f"  k=N ({kN}): {accN:.4f}")
print(f"  k={k_valid:<7}: {accv:.4f}")

print("\nConfusion matrix (final k):")
print(confusion_matrix(y_test, pv, labels=labels))

print("\nClassification report (final k):")
print(classification_report(y_test, pv, labels=labels, target_names=target_names))


# -----------------------------
# 2D Decision-region plots (honest: model trained ONLY on 2D slice)
# -----------------------------
# def plot_decision_regions_2d(model2d, X2_train, X2_test, y_train, y_test, title: str):
#     pad = 0.25
#     x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
#     y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad

#     xx, yy = np.meshgrid(
#         np.linspace(x_min, x_max, 600),
#         np.linspace(y_min, y_max, 600)
#     )
#     grid2 = np.c_[xx.ravel(), yy.ravel()]
#     Z = model2d.predict(grid2).reshape(xx.shape)

#     plt.figure(figsize=(7, 5))
#     levels = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
#     plt.contourf(xx, yy, Z, alpha=0.25, levels=levels)

#     for c in labels:
#         plt.scatter(
#             X2_train[y_train == c, 0], X2_train[y_train == c, 1],
#             s=14, label=f"{c}: {label_names[c]} (train)"
#         )
#     for c in labels:
#         plt.scatter(
#             X2_test[y_test == c, 0], X2_test[y_test == c, 1],
#             s=35, marker="x", label=f"{c}: {label_names[c]} (test)"
#         )

#     plt.xlabel(f"{PLOT_X_LABEL} (scaled)")
#     plt.ylabel(f"{PLOT_Y_LABEL} (scaled)")
#     plt.title(title)
#     plt.legend(loc="best", fontsize=8, ncol=2)
#     plt.tight_layout()
#     plt.show()


# def fit_2d_knn(k: int):
#     m = KNeighborsClassifier(n_neighbors=k, weights="distance")
#     m.fit(X2_train, y_train)
#     return m

# m1_2d = fit_2d_knn(k1)
# mN_2d = fit_2d_knn(kN)
# mv_2d = fit_2d_knn(k_valid)

# plot_decision_regions_2d(m1_2d, X2_train, X2_test, y_train, y_test, "2D decision regions (k=1)")
# plot_decision_regions_2d(mN_2d, X2_train, X2_test, y_train, y_test, f"2D decision regions (k={kN})")
# plot_decision_regions_2d(mv_2d, X2_train, X2_test, y_train, y_test, f"2D decision regions (k={k_valid})")

# -----------------------------
# 2D Decision-region plots (SAVE ONLY, do not show)
# -----------------------------

# Use ONLY the first two features for plotting
X2 = X[:, [0, 1]].copy()

# Split consistently
idx_all = np.arange(len(X2))
idx_train2, idx_test2 = train_test_split(
    idx_all,
    test_size=0.10,
    random_state=0,
    stratify=y
)

X2_train = X2[idx_train2]
X2_test  = X2[idx_test2]
y_train2 = y[idx_train2]
y_test2  = y[idx_test2]

# Scale 2D space
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test  = scaler2.transform(X2_test)


def fit_2d_knn(k: int):
    model2d = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model2d.fit(X2_train, y_train2)
    return model2d


def save_decision_region_plot(model2d, filename: str, title: str):
    pad = 0.25
    x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
    y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 600),
        np.linspace(y_min, y_max, 600)
    )
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    Z = model2d.predict(grid2).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    levels = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    plt.contourf(xx, yy, Z, alpha=0.25, levels=levels)

    for c in labels:
        plt.scatter(
            X2_train[y_train2 == c, 0], X2_train[y_train2 == c, 1],
            s=14, label=f"{c}: {label_names[c]} (train)"
        )
    for c in labels:
        plt.scatter(
            X2_test[y_test2 == c, 0], X2_test[y_test2 == c, 1],
            s=35, marker="x", label=f"{c}: {label_names[c]} (test)"
        )

    plt.xlabel(f"{FEATURE_SET[0]} (scaled)")
    plt.ylabel(f"{FEATURE_SET[1]} (scaled)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()  # Important: prevent display


# Train 2D models
m1_2d = fit_2d_knn(k1)
mN_2d = fit_2d_knn(kN)
mv_2d = fit_2d_knn(k_valid)

# Save 3 plots
save_decision_region_plot(m1_2d, "knn_k1.png", f"k-NN decision regions (k=1)")
save_decision_region_plot(mN_2d, f"knn_k{kN}.png", f"k-NN decision regions (k={kN})")
save_decision_region_plot(mv_2d, f"knn_k{k_valid}.png", f"k-NN decision regions (k={k_valid})")

print("\nSaved plots:")
print("  knn_k1.png")
print(f"  knn_k{kN}.png")
print(f"  knn_k{k_valid}.png")

def save_decision_region_plot_2d(model2d, X2_train, X2_test, y_train, y_test, filename: str, title: str):
    pad = 0.25
    x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
    y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 600),
        np.linspace(y_min, y_max, 600)
    )
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    Z = model2d.predict(grid2).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
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
# -----------------------------
# PCA plots (2D) - SAVE ONLY
# -----------------------------
def fit_knn_on_2d(X2_train, y_train, k: int):
    m = KNeighborsClassifier(n_neighbors=k, weights="distance")
    m.fit(X2_train, y_train)
    return m

m1_pca = fit_knn_on_2d(X_train_pca, y_train, k1)
mN_pca = fit_knn_on_2d(X_train_pca, y_train, kN)
mv_pca = fit_knn_on_2d(X_train_pca, y_train, k_valid)

save_decision_region_plot_2d(m1_pca, X_train_pca, X_test_pca, y_train, y_test,
                             "pca_knn_k1.png", "PCA decision regions (k=1)")

save_decision_region_plot_2d(mN_pca, X_train_pca, X_test_pca, y_train, y_test,
                             f"pca_knn_k{kN}.png", f"PCA decision regions (k={kN})")

save_decision_region_plot_2d(mv_pca, X_train_pca, X_test_pca, y_train, y_test,
                             f"pca_knn_k{k_valid}.png", f"PCA decision regions (k={k_valid})")

print("\nSaved PCA plots:")
print("  pca_knn_k1.png")
print(f"  pca_knn_k{kN}.png")
print(f"  pca_knn_k{k_valid}.png")
