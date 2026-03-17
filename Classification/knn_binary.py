# Task: K-nn for Dynamic vs Static classification :-) :-) :-D
# Features:
#   - tBodyAccMag-mean()  (feature #201 in features.txt)
#   - tBodyAccMag-std()   (feature #202 in features.txt)
# Labels:
#   Dynamic (1): {1,2,3} = walking, upstairs, downstairs
#   Static  (0): {4,5,6} = sitting, standing, laying

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Data
X_TRAIN_PATH = "X_train.txt"
Y_TRAIN_PATH = "y_train.txt"
FEATURES_PATH = "features.txt"

# load features
def load_feature_names(features_path: str) -> list[str]:
    names = []
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                names.append(parts[1])
    return names

feature_names = load_feature_names(FEATURES_PATH)
FEAT_A = "tBodyAccMag-mean()"
FEAT_B = "tBodyAccMag-std()"

try:
    idx_a_0based = feature_names.index(FEAT_A)
    idx_b_0based = feature_names.index(FEAT_B)
except ValueError as e:
    raise ValueError(
        f"FEATURE PROBLEM!!!!!!!!!!!!!"
    ) from e
# print(f"Using features:")
# print(f"  {FEAT_A} -> column {idx_a_0based+1} (1-based)")
# print(f"  {FEAT_B} -> column {idx_b_0based+1} (1-based)")


# load X_train and y_train
X_full = np.loadtxt(X_TRAIN_PATH)
y_activity = np.loadtxt(Y_TRAIN_PATH, dtype=int)

X = X_full[:, [idx_a_0based, idx_b_0based]].astype(float)

# 6 labels --> binary
# Dynamic: 1,2,3 
# Static: 4,5,6
y = np.zeros_like(y_activity, dtype=int)
y[np.isin(y_activity, [1, 2, 3])] = 1
y[np.isin(y_activity, [4, 5, 6])] = 0

#choose only 100 (Half hald in this case for the two labels)
np.random.seed(0)
N_PER_CLASS = 50
idx_static = np.where(y == 0)[0]
idx_dynamic = np.where(y == 1)[0]
idx_static_sel = np.random.choice(idx_static, size=N_PER_CLASS, replace=False)
idx_dynamic_sel = np.random.choice(idx_dynamic, size=N_PER_CLASS, replace=False)
idx = np.concatenate([idx_static_sel, idx_dynamic_sel])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
print(f"Using {len(X)} samples")
n_dyn = int(np.sum(y == 1))
n_sta = int(np.sum(y == 0))
print(f"\nCounts: dynamic={n_dyn}, static={n_sta}")


# 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=0,
    stratify=y
)
N = len(X_train)
k1 = 1
kN = N

# Choose a k
# sqrt(N) here for Q3
# make it odd!!!!!!! ODD!!!!!!!!!!!!!
k_valid = int(np.sqrt(N))
if k_valid % 2 == 0:
    k_valid += 1
k_valid = max(3, k_valid)
print(f"\nN_train = {N}")
print(f"k choices: k=1, k=N={kN}, k_valid={k_valid}")


# Q4
# fit + evaluate
def fit_and_eval(k: int):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return model, pred, acc
m1, p1, acc1 = fit_and_eval(k1)
mN, pN, accN = fit_and_eval(kN)
mv, pv, accv = fit_and_eval(k_valid)
print("\nHoldout (10%) accuracy:")
print(f"  k=1       : {acc1:.4f}")
print(f"  k=N ({kN}): {accN:.4f}")
print(f"  k={k_valid:<7}: {accv:.4f}")
print("\nConfusion matrix (final k):")
print(confusion_matrix(y_test, pv))
print("\nClassification report (final k):")
print(classification_report(y_test, pv, target_names=["static", "dynamic"]))


# plot
def plot_decision_regions(model, title: str):
    pad = 0.10
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 600),
        np.linspace(y_min, y_max, 600)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5])

    # Train points
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], s=14, label="static (train)")
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], s=14, label="dynamic (train)")

    # Test points
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], s=35, marker="x", label="static (test)")
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], s=35, marker="x", label="dynamic (test)")

    plt.xlabel(FEAT_A)
    plt.ylabel(FEAT_B)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

plot_decision_regions(m1, "k-NN decision regions (k=1)")
plot_decision_regions(mN, f"k-NN decision regions (k=N={kN})")
plot_decision_regions(mv, f"k-NN decision regions (k={k_valid})")


# # 6) Neighbors for one example point
# # Pick one test point and show its nearest neighbors under the final model
# i0 = 0
# x_star = X_test[i0:i0+1]
# dists, idxs = mv.kneighbors(x_star, n_neighbors=min(k_valid, 15), return_distance=True)
# print("\nExample x* (first test sample):", x_star.ravel().tolist())
# print("Nearest neighbors (up to 15):")
# for r, (di, ii) in enumerate(zip(dists[0], idxs[0]), start=1):
#     print(f"  {r:2d}) dist={di:.4f}, neighbor_label={'dynamic' if y_train[ii]==1 else 'static'}")
