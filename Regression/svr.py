import os
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ---------------------------
# User settings (EDIT HERE)
# ---------------------------
DATA_DIR = r"./slow walking/data1/syn"   # <-- point to your syn folder
TRAIN_IDS = [1]
TEST_IDS  = [7]

IMU_FILE_FMT = "imu{}.csv"
VI_FILE_FMT  = "vi{}.csv"

# Column mapping (0-based)
IMU_T_COL = 0
IMU_ACC_COLS = [1, 2, 3]      # ax ay az
IMU_GYR_COLS = [4, 5, 6]      # gx gy gz
IMU_USE_COLS = IMU_ACC_COLS + IMU_GYR_COLS

VI_T_COL = 0
VI_VEL_COLS = [5, 6, 7]       # vx vy vz  (based on your vi1.csv sample)

# Feature construction
N_LAGS = 20                   # number of IMU samples in the past to stack (tune this!)
MIN_PAIRS_PER_SEQ = 200       # skip a sequence if too few aligned pairs


# ---------------------------
# Helpers
# ---------------------------
def read_csv_noheader(path: str) -> pd.DataFrame:
    # Your files appear to have NO header row (numbers in first row),
    # so we force header=None.
    return pd.read_csv(path, header=None)

def infer_time_scale_to_seconds(t: np.ndarray) -> np.ndarray:
    """
    Convert timestamps to seconds with a simple heuristic based on median dt.
    Works for ns/us/ms/s-ish logs.
    """
    t = np.asarray(t, dtype=np.float64)
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return t - t[0]

    # Heuristic:
    # ns -> dt ~ 1e6..1e8 (for 100-1000Hz) if timestamps are ns-ish big
    # us -> dt ~ 1e3..1e6
    # ms -> dt ~ 1..1e3
    # s  -> dt ~ 1e-3..1
    if dt > 1e6:
        t_sec = t / 1e9
    elif dt > 1e3:
        t_sec = t / 1e6
    elif dt > 1:
        t_sec = t / 1e3
    else:
        t_sec = t
    return t_sec - t_sec[0]

def build_xy_for_sequence(imu_df: pd.DataFrame, vi_df: pd.DataFrame,
                          n_lags: int,
                          imu_use_cols: list[int],
                          imu_t_col: int,
                          vi_t_col: int,
                          vi_vel_cols: list[int]):
    """
    For each VI timestamp, find the latest IMU sample at/before that time,
    then stack the previous n_lags IMU rows as features.
    """
    imu_t = infer_time_scale_to_seconds(imu_df.iloc[:, imu_t_col].to_numpy())
    vi_t  = infer_time_scale_to_seconds(vi_df.iloc[:, vi_t_col].to_numpy())

    imu_sig = imu_df.iloc[:, imu_use_cols].to_numpy(dtype=np.float64)
    vi_vel  = vi_df.iloc[:, vi_vel_cols].to_numpy(dtype=np.float64)

    # Ensure sorted by time
    imu_order = np.argsort(imu_t)
    vi_order  = np.argsort(vi_t)

    imu_t = imu_t[imu_order]
    imu_sig = imu_sig[imu_order]

    vi_t = vi_t[vi_order]
    vi_vel = vi_vel[vi_order]

    X_list, y_list = [], []

    # For each VI time, find IMU index i s.t. imu_t[i] <= vi_t[k] < imu_t[i+1]
    imu_indices = np.searchsorted(imu_t, vi_t, side="right") - 1

    for k, i in enumerate(imu_indices):
        if i < n_lags - 1:
            continue
        if i >= len(imu_t):
            continue

        # Stack last n_lags samples
        window = imu_sig[i - (n_lags - 1): i + 1, :]   # shape (n_lags, n_feat)
        X_list.append(window.reshape(-1))              # flatten
        y_list.append(vi_vel[k])

    if not X_list:
        return None, None

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y

def load_sequence(seq_id: int, data_dir: str):
    imu_path = os.path.join(data_dir, IMU_FILE_FMT.format(seq_id))
    vi_path  = os.path.join(data_dir, VI_FILE_FMT.format(seq_id))

    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"Missing IMU file: {imu_path}")
    if not os.path.exists(vi_path):
        raise FileNotFoundError(f"Missing VI file: {vi_path}")

    imu_df = read_csv_noheader(imu_path)
    vi_df  = read_csv_noheader(vi_path)
    return imu_df, vi_df


# ---------------------------
# Main
# ---------------------------
def main():
    # Build train set
    X_tr_all, y_tr_all = [], []
    for sid in TRAIN_IDS:
        imu_df, vi_df = load_sequence(sid, DATA_DIR)
        X, y = build_xy_for_sequence(
            imu_df, vi_df,
            n_lags=N_LAGS,
            imu_use_cols=IMU_USE_COLS,
            imu_t_col=IMU_T_COL,
            vi_t_col=VI_T_COL,
            vi_vel_cols=VI_VEL_COLS
        )
        if X is None or len(X) < MIN_PAIRS_PER_SEQ:
            print(f"[WARN] seq {sid}: too few pairs, skipping.")
            continue
        X_tr_all.append(X)
        y_tr_all.append(y)
        print(f"[OK] train seq {sid}: X={X.shape}, y={y.shape}")

    X_train = np.vstack(X_tr_all)
    y_train = np.vstack(y_tr_all)
    print(f"\nTrain total: X={X_train.shape}, y={y_train.shape}")

    # Model: scale X and y, MultiOutput SVR
    # (Scale y by wrapping estimator; easiest: scale y manually)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    y_train_s = y_scaler.fit_transform(y_train)

    base_svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01)
    model = MultiOutputRegressor(base_svr, n_jobs=-1)
    model.fit(X_train_s, y_train_s)
    print("\n[OK] Model trained.")

    # Evaluate on test sequences
    for sid in TEST_IDS:
        imu_df, vi_df = load_sequence(sid, DATA_DIR)
        X_te, y_te = build_xy_for_sequence(
            imu_df, vi_df,
            n_lags=N_LAGS,
            imu_use_cols=IMU_USE_COLS,
            imu_t_col=IMU_T_COL,
            vi_t_col=VI_T_COL,
            vi_vel_cols=VI_VEL_COLS
        )
        if X_te is None or len(X_te) < MIN_PAIRS_PER_SEQ:
            print(f"[WARN] seq {sid}: too few pairs, skipping eval.")
            continue

        X_te_s = x_scaler.transform(X_te)
        y_pred_s = model.predict(X_te_s)

        y_pred = y_scaler.inverse_transform(y_pred_s)
        y_true = y_te

        rmse_xyz = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
        rmse_speed = np.sqrt(mean_squared_error(
            np.linalg.norm(y_true, axis=1),
            np.linalg.norm(y_pred, axis=1)
        ))

        print(f"\n=== Test seq {sid} ===")
        print(f"RMSE vx,vy,vz: {rmse_xyz}")
        print(f"RMSE speed   : {rmse_speed:.6f}")

    print("\nDone.")

if __name__ == "__main__":
    main()