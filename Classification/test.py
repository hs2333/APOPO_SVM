import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ==========================
# CONFIG
# ==========================
DATA_FOLDER = "slow walking/data1/syn"
SEQ_ID = 1  # use imu1 & vi1 for demonstration
HEADER_TIME_SCALE = 1e-3
G_TO_MPS2 = 9.80665


# ==========================
# Load IMU
# ==========================
imu = pd.read_csv(f"{DATA_FOLDER}/imu{SEQ_ID}.csv", header=None)
imu.columns = [
    "Time",
    "attitude_roll",
    "attitude_pitch",
    "attitude_yaw",
    "rotation_rate_x",
    "rotation_rate_y",
    "rotation_rate_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "user_acc_x",
    "user_acc_y",
    "user_acc_z",
    "magnetic_field_x",
    "magnetic_field_y",
    "magnetic_field_z",
]
imu = imu.apply(pd.to_numeric, errors="coerce").dropna()

# convert G → m/s²
ax = imu["user_acc_x"].to_numpy() * G_TO_MPS2


# ==========================
# Load Vicon
# ==========================
vi = pd.read_csv(f"{DATA_FOLDER}/vi{SEQ_ID}.csv", header=None)
vi.columns = [
    "Time",
    "Header",
    "translation.x",
    "translation.y",
    "translation.z",
    "rotation.x",
    "rotation.y",
    "rotation.z",
    "rotation.w",
]
vi = vi.apply(pd.to_numeric, errors="coerce").dropna()

t = vi["Header"].to_numpy() * HEADER_TIME_SCALE
x = vi["translation.x"].to_numpy()

# compute velocity from position
dt = np.diff(t)
dx = np.diff(x)

valid = dt > 0
vx = dx[valid] / dt[valid]

# align ax with vx
ax = ax[:-1][valid]


# ==========================
# Linear Regression
# ==========================
X = ax.reshape(-1, 1)
y = vx

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("Intercept:", model.intercept_)
print("Slope (theta1):", model.coef_[0])
print("MSE:", mean_squared_error(y, y_pred))


# ==========================
# Plot
# ==========================
plt.figure(figsize=(8,6))

# scatter data
plt.scatter(ax, vx, s=5, alpha=0.4, label="Data")

# regression line
ax_sorted = np.linspace(ax.min(), ax.max(), 200)
vx_line = model.intercept_ + model.coef_[0] * ax_sorted
plt.plot(ax_sorted, vx_line, color="red", linewidth=2, label="Linear model")

plt.xlabel("IMU user_acc_x (m/s²)")
plt.ylabel("Ground Truth v_x (m/s)")
plt.title("Linear Regression: v_x vs a_x")
plt.legend()
plt.tight_layout()
plt.show()
