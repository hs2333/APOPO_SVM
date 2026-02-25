"""
Dataset: UCI Concrete Compressive Strength (id=165) via ucimlrepo

Tasks covered:
1) Load dataset + link + explain features (print metadata/variables)
2) Linear regression with Scikit-Learn + plot + train MSE
3) Linear regression via Normal Equation + compare to sklearn
4) Evaluate both on test set + plot + test MSE + compare train vs test
5) Brief printed discussion prompts (you write these in report)

Notes:
- Uses 80% training / 20% testing
- Uses ONE feature for the main "data + line" plot (because multi-feature regression
  doesn't have a single line to overlay cleanly).
- Still trains the REAL multi-feature model for performance reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# data
# from "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength"
concrete = fetch_ucirepo(id=165)
X_df: pd.DataFrame = concrete.data.features
y_df: pd.DataFrame = concrete.data.targets
y = y_df.iloc[:, 0].to_numpy(dtype=float)
X = X_df.to_numpy(dtype=float)

# print(concrete.metadata)
# print(concrete.variables)

# print("\nShapes:")
# print("X:", X.shape, "y:", y.shape)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# scikit
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

yhat_train_sk = sk_model.predict(X_train)
yhat_test_sk = sk_model.predict(X_test)

mse_train_sk = mean_squared_error(y_train, yhat_train_sk)
mse_test_sk = mean_squared_error(y_test, yhat_test_sk)

print("\nScikit-Learn LinearRegression")
print("θ0:", sk_model.intercept_)
print("Coefficients (θ1-θ8):")
for name, coef in zip(X_df.columns, sk_model.coef_):
    print(f"  {name:30s} {coef: .6f}")
print(f"Train MSE: {mse_train_sk:.6f}")
print(f"Test  MSE: {mse_test_sk:.6f}")

# normal equation
def normal_equation_fit(X_in: np.ndarray, y_in: np.ndarray) -> np.ndarray:
    Xb = np.c_[np.ones((X_in.shape[0], 1)), X_in]
    theta = np.linalg.pinv(Xb) @ y_in
    return theta

theta_ne = normal_equation_fit(X_train, y_train)

# predict with theta
Xb_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
Xb_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

yhat_train_ne = Xb_train @ theta_ne
yhat_test_ne = Xb_test @ theta_ne

mse_train_ne = mean_squared_error(y_train, yhat_train_ne)
mse_test_ne = mean_squared_error(y_test, yhat_test_ne)

print("\nNormal Equation")
print("θ0:", theta_ne[0])
print("θ1-θ8:")
for name, coef in zip(X_df.columns, theta_ne[1:]):
    print(f"  {name:30s} {coef: .6f}")
print(f"Train MSE: {mse_train_ne:.6f}")
print(f"Test  MSE: {mse_test_ne:.6f}")

# print("\nSklearn vs Normal: differences")
# print("Δθ0:", abs(sk_model.intercept_ - theta_ne[0]))
# print("max |Δθi|:", np.max(np.abs(sk_model.coef_ - theta_ne[1:])))

# plot
FEATURE_FOR_PLOT = X_df.columns[0]  # 0 is concrete
x1 = X_df[FEATURE_FOR_PLOT].to_numpy(dtype=float).reshape(-1, 1)

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y, test_size=0.2, random_state=42, shuffle=True
)

sk_1d = LinearRegression()
sk_1d.fit(x1_train, y1_train)

y1_train_pred = sk_1d.predict(x1_train)
y1_test_pred = sk_1d.predict(x1_test)

mse_1d_train = mean_squared_error(y1_train, y1_train_pred)
mse_1d_test = mean_squared_error(y1_test, y1_test_pred)

print("Model: y = θ0 + θ1*x")
print("θ0:", sk_1d.intercept_)
print("θ1:", sk_1d.coef_[0])
print(f"Train MSE (1D): {mse_1d_train:.6f}")
print(f"Test  MSE (1D): {mse_1d_test:.6f}")

# plot: scatter + line 
plt.figure()
plt.scatter(x1_train, y1_train, s=18, alpha=0.6)
x_line = np.linspace(x1_train.min(), x1_train.max(), 200).reshape(-1, 1)
y_line = sk_1d.predict(x_line)

plt.plot(x_line, y_line, linewidth=2)
plt.xlabel(FEATURE_FOR_PLOT)
plt.ylabel("Concrete compressive strength (MPa)")
plt.title("Concrete Strength: Linear Regression")
plt.tight_layout()
plt.show()

# plot: test set + the same line
plt.figure()
plt.scatter(x1_test, y1_test, s=18, alpha=0.6)
plt.plot(x_line, y_line, linewidth=2)
plt.xlabel(FEATURE_FOR_PLOT)
plt.ylabel("Concrete compressive strength (MPa)")
plt.title("Test Set: Data + Linear Model")
plt.tight_layout()
plt.show()

# # =========================
# # 6) Optional: Predicted vs True plot for test (multi-feature)
# #    (Not required, but very useful for your discussion)
# # =========================
# plt.figure()
# plt.scatter(y_test, yhat_test_sk, s=18, alpha=0.6)
# plt.xlabel("True strength (MPa)")
# plt.ylabel("Predicted strength (MPa)")
# plt.title("Multi-feature Model: True vs Predicted (Test Set)")
# plt.tight_layout()
# plt.show()
