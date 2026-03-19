import os
import time
import json
import pickle
import platform
import tracemalloc

import numpy as np
import pandas as pd
import psutil
import sklearn

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =============================
# User settings
# =============================
DATA_DIR = r"./running/data1/syn"
TRAIN_IDS = [1, 2, 3, 4, 5, 6]
TEST_IDS  = [7]

IMU_FILE_FMT = "imu{}.csv"
VI_FILE_FMT  = "vi{}.csv"

# Column mapping (0-based)
IMU_T_COL = 0
IMU_ACC_COLS = [1, 2, 3]      # ax ay az
IMU_GYR_COLS = [4, 5, 6]      # gx gy gz
IMU_USE_COLS = IMU_ACC_COLS + IMU_GYR_COLS

VI_T_COL = 0
VI_VEL_COLS = [5, 6]          # vx vy ONLY (ignore z)

# Feature construction
N_LAGS = 20
MIN_PAIRS_PER_SEQ = 200

# Training / model settings
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_GAMMA = "scale"
SVR_EPSILON = 0.01
MAX_TRAIN = 20000

# Benchmark settings
WARMUP_RUNS = 5
TIMED_RUNS_SINGLE = 50
TIMED_RUNS_BATCH = 10

# Output files
MODEL_PATH = "svr_multioutput_model.pkl"
REPORT_PATH = "svr_benchmark_report.txt"
JSON_REPORT_PATH = "svr_benchmark_report.json"


# =============================
# Helpers
# =============================
def read_csv_noheader(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None)

def bytes_to_mb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 2)

def bytes_to_gb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)

def infer_time_scale_to_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return t - t[0]

    if dt > 1e6:
        t_sec = t / 1e9
    elif dt > 1e3:
        t_sec = t / 1e6
    elif dt > 1:
        t_sec = t / 1e3
    else:
        t_sec = t
    return t_sec - t_sec[0]

def build_xy_for_sequence(
    imu_df: pd.DataFrame,
    vi_df: pd.DataFrame,
    n_lags: int,
    imu_use_cols: list[int],
    imu_t_col: int,
    vi_t_col: int,
    vi_vel_cols: list[int]
):
    imu_t = infer_time_scale_to_seconds(imu_df.iloc[:, imu_t_col].to_numpy())
    vi_t  = infer_time_scale_to_seconds(vi_df.iloc[:, vi_t_col].to_numpy())

    imu_sig = imu_df.iloc[:, imu_use_cols].to_numpy(dtype=np.float64)
    vi_vel  = vi_df.iloc[:, vi_vel_cols].to_numpy(dtype=np.float64)  # shape (N,2)

    imu_order = np.argsort(imu_t)
    vi_order  = np.argsort(vi_t)

    imu_t = imu_t[imu_order]
    imu_sig = imu_sig[imu_order]

    vi_t = vi_t[vi_order]
    vi_vel = vi_vel[vi_order]

    X_list, y_list = [], []
    imu_indices = np.searchsorted(imu_t, vi_t, side="right") - 1

    for k, i in enumerate(imu_indices):
        if i < n_lags - 1:
            continue
        if i >= len(imu_t):
            continue

        window = imu_sig[i - (n_lags - 1): i + 1, :]
        X_list.append(window.reshape(-1))
        y_list.append(vi_vel[k])

    if not X_list:
        return None, None

    X = np.vstack(X_list)
    y = np.vstack(y_list)   # shape (N,2)
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

def get_system_info() -> dict:
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz_max": cpu_freq.max if cpu_freq is not None else None,
        "cpu_freq_mhz_current": cpu_freq.current if cpu_freq is not None else None,
        "total_ram_gb": bytes_to_gb(vm.total),
        "available_ram_gb": bytes_to_gb(vm.available),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }

def get_process_memory_info() -> dict:
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": bytes_to_mb(mem.rss),
        "vms_mb": bytes_to_mb(mem.vms),
    }

def benchmark_single_sample_latency(model, x_single, warmup_runs=50, timed_runs=500):
    for _ in range(warmup_runs):
        model.predict(x_single)

    latencies_ms = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        model.predict(x_single)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    throughput_inf_per_sec = 1000.0 / avg_ms if avg_ms > 0 else float("inf")

    return {
        "avg_latency_ms": avg_ms,
        "median_latency_ms": median_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "p95_latency_ms": p95_ms,
        "throughput_inf_per_sec_from_avg_latency": throughput_inf_per_sec,
        "num_runs": timed_runs,
    }

def benchmark_batch_throughput(model, X_batch, warmup_runs=50, timed_runs=100):
    n_samples = len(X_batch)

    for _ in range(warmup_runs):
        model.predict(X_batch)

    times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        model.predict(X_batch)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_sec = float(np.mean(times))
    median_sec = float(np.median(times))
    batches_per_sec = 1.0 / avg_sec if avg_sec > 0 else float("inf")
    samples_per_sec = n_samples / avg_sec if avg_sec > 0 else float("inf")

    return {
        "batch_size": n_samples,
        "avg_batch_time_ms": avg_sec * 1000.0,
        "median_batch_time_ms": median_sec * 1000.0,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "num_runs": timed_runs,
    }

def save_text_report(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =============================
# Main
# =============================
def main():
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

    if not X_tr_all:
        raise RuntimeError("No training sequences available after filtering.")

    X_train = np.vstack(X_tr_all)
    y_train = np.vstack(y_tr_all)
    print(f"\nTrain total: X={X_train.shape}, y={y_train.shape}")

    mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]
    print("After cleaning:", X_train.shape)

    if len(X_train) > MAX_TRAIN:
        step = max(1, len(X_train) // MAX_TRAIN)
        X_train = X_train[::step]
        y_train = y_train[::step]
        print("Subsampled:", X_train.shape)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    y_train_s = y_scaler.fit_transform(y_train)

    base_svr = SVR(
        kernel=SVR_KERNEL,
        C=SVR_C,
        gamma=SVR_GAMMA,
        epsilon=SVR_EPSILON
    )
    model = MultiOutputRegressor(base_svr, n_jobs=1)

    # -------------------------
    # Training benchmark
    # -------------------------
    tracemalloc.start()
    mem_before_train = get_process_memory_info()

    train_t0 = time.perf_counter()
    model.fit(X_train_s, y_train_s)
    train_t1 = time.perf_counter()

    mem_after_train = get_process_memory_info()
    current_mem_bytes, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    training_time_sec = train_t1 - train_t0

    print("\n[OK] Model trained.")
    print(f"Training time: {training_time_sec:.4f} s")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "settings": {
                    "n_lags": N_LAGS,
                    "imu_use_cols": IMU_USE_COLS,
                    "vi_vel_cols": VI_VEL_COLS,
                    "kernel": SVR_KERNEL,
                    "C": SVR_C,
                    "gamma": SVR_GAMMA,
                    "epsilon": SVR_EPSILON,
                },
            },
            f
        )

    model_size_mb = bytes_to_mb(os.path.getsize(MODEL_PATH))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Model size: {model_size_mb:.4f} MB")

    # -------------------------
    # Evaluation + benchmark
    # -------------------------
    system_info = get_system_info()
    runtime_mem = get_process_memory_info()

    all_test_results = []
    benchmark_reference_X = None

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

        rmse_vxvy = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
        mae_vxvy = np.mean(np.abs(y_pred - y_true), axis=0)
        r2_vxvy = [
            r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
        ]

        rmse_speed_2d = np.sqrt(mean_squared_error(
            np.linalg.norm(y_true, axis=1),
            np.linalg.norm(y_pred, axis=1)
        ))

        print(f"\n=== Test seq {sid} ===")
        print(f"RMSE vx,vy     : {rmse_vxvy}")
        print(f"MAE  vx,vy     : {mae_vxvy}")
        print(f"R2   vx,vy     : {r2_vxvy}")
        print(f"RMSE speed2D   : {rmse_speed_2d:.6f}")

        all_test_results.append({
            "seq_id": sid,
            "num_samples": int(len(X_te)),
            "rmse_vx": float(rmse_vxvy[0]),
            "rmse_vy": float(rmse_vxvy[1]),
            "mae_vx": float(mae_vxvy[0]),
            "mae_vy": float(mae_vxvy[1]),
            "r2_vx": float(r2_vxvy[0]),
            "r2_vy": float(r2_vxvy[1]),
            "rmse_speed_2d": float(rmse_speed_2d),
        })

        if benchmark_reference_X is None and len(X_te_s) > 0:
            benchmark_reference_X = X_te_s

    if benchmark_reference_X is None:
        raise RuntimeError("No usable test data available for benchmarking.")

    # -------------------------
    # Inference benchmark
    # -------------------------
    x_single = benchmark_reference_X[0:1]

    single_bench = benchmark_single_sample_latency(
        model,
        x_single,
        warmup_runs=WARMUP_RUNS,
        timed_runs=TIMED_RUNS_SINGLE
    )

    batch_bench = benchmark_batch_throughput(
        model,
        benchmark_reference_X,
        warmup_runs=WARMUP_RUNS,
        timed_runs=TIMED_RUNS_BATCH
    )

    print("\nSingle-sample inference benchmark:")
    print(f"  Avg latency   : {single_bench['avg_latency_ms']:.6f} ms")
    print(f"  Median latency: {single_bench['median_latency_ms']:.6f} ms")
    print(f"  P95 latency   : {single_bench['p95_latency_ms']:.6f} ms")
    print(f"  Min latency   : {single_bench['min_latency_ms']:.6f} ms")
    print(f"  Max latency   : {single_bench['max_latency_ms']:.6f} ms")
    print(f"  Throughput    : {single_bench['throughput_inf_per_sec_from_avg_latency']:.2f} inf/sec")

    print("\nBatch inference benchmark:")
    print(f"  Batch size        : {batch_bench['batch_size']}")
    print(f"  Avg batch time    : {batch_bench['avg_batch_time_ms']:.6f} ms")
    print(f"  Median batch time : {batch_bench['median_batch_time_ms']:.6f} ms")
    print(f"  Batches/sec       : {batch_bench['batches_per_sec']:.2f}")
    print(f"  Samples/sec       : {batch_bench['samples_per_sec']:.2f}")

    print("\nSystem / hardware info:")
    for k, v in system_info.items():
        print(f"  {k}: {v}")

    print("\nProcess memory usage:")
    print(f"  Before training RSS : {mem_before_train['rss_mb']:.3f} MB")
    print(f"  After training RSS  : {mem_after_train['rss_mb']:.3f} MB")
    print(f"  Current process RSS : {runtime_mem['rss_mb']:.3f} MB")
    print(f"  Peak traced memory  : {bytes_to_mb(peak_mem_bytes):.3f} MB")

    # -------------------------
    # Save reports
    # -------------------------
    report_lines = []
    report_lines.append("SVR Benchmark Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("Model settings:")
    report_lines.append(f"  kernel: {SVR_KERNEL}")
    report_lines.append(f"  C: {SVR_C}")
    report_lines.append(f"  gamma: {SVR_GAMMA}")
    report_lines.append(f"  epsilon: {SVR_EPSILON}")
    report_lines.append(f"  n_lags: {N_LAGS}")
    report_lines.append(f"  imu_use_cols: {IMU_USE_COLS}")
    report_lines.append(f"  train_ids: {TRAIN_IDS}")
    report_lines.append(f"  test_ids: {TEST_IDS}")
    report_lines.append(f"  max_train: {MAX_TRAIN}")
    report_lines.append("")

    report_lines.append("Training:")
    report_lines.append(f"  X_train shape: {tuple(X_train.shape)}")
    report_lines.append(f"  y_train shape: {tuple(y_train.shape)}")
    report_lines.append(f"  Training time (s): {training_time_sec:.6f}")
    report_lines.append(f"  Saved model size (MB): {model_size_mb:.6f}")
    report_lines.append("")

    report_lines.append("Per-sequence regression performance:")
    for result in all_test_results:
        report_lines.append(f"  Test seq {result['seq_id']}:")
        report_lines.append(f"    num_samples: {result['num_samples']}")
        report_lines.append(f"    rmse_vx: {result['rmse_vx']:.6f}")
        report_lines.append(f"    rmse_vy: {result['rmse_vy']:.6f}")
        report_lines.append(f"    mae_vx: {result['mae_vx']:.6f}")
        report_lines.append(f"    mae_vy: {result['mae_vy']:.6f}")
        report_lines.append(f"    r2_vx: {result['r2_vx']:.6f}")
        report_lines.append(f"    r2_vy: {result['r2_vy']:.6f}")
        report_lines.append(f"    rmse_speed_2d: {result['rmse_speed_2d']:.6f}")
    report_lines.append("")

    report_lines.append("Single-sample inference benchmark:")
    report_lines.append(f"  Average latency (ms): {single_bench['avg_latency_ms']:.6f}")
    report_lines.append(f"  Median latency (ms): {single_bench['median_latency_ms']:.6f}")
    report_lines.append(f"  P95 latency (ms): {single_bench['p95_latency_ms']:.6f}")
    report_lines.append(f"  Min latency (ms): {single_bench['min_latency_ms']:.6f}")
    report_lines.append(f"  Max latency (ms): {single_bench['max_latency_ms']:.6f}")
    report_lines.append(f"  Throughput (inf/sec): {single_bench['throughput_inf_per_sec_from_avg_latency']:.6f}")
    report_lines.append("")

    report_lines.append("Batch inference benchmark:")
    report_lines.append(f"  Batch size: {batch_bench['batch_size']}")
    report_lines.append(f"  Avg batch time (ms): {batch_bench['avg_batch_time_ms']:.6f}")
    report_lines.append(f"  Median batch time (ms): {batch_bench['median_batch_time_ms']:.6f}")
    report_lines.append(f"  Batches/sec: {batch_bench['batches_per_sec']:.6f}")
    report_lines.append(f"  Samples/sec: {batch_bench['samples_per_sec']:.6f}")
    report_lines.append("")

    report_lines.append("System / hardware info:")
    for k, v in system_info.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")

    report_lines.append("Memory:")
    report_lines.append(f"  Process RSS before training (MB): {mem_before_train['rss_mb']:.6f}")
    report_lines.append(f"  Process RSS after training (MB): {mem_after_train['rss_mb']:.6f}")
    report_lines.append(f"  Current process RSS (MB): {runtime_mem['rss_mb']:.6f}")
    report_lines.append(f"  Peak traced memory (MB): {bytes_to_mb(peak_mem_bytes):.6f}")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    save_text_report(REPORT_PATH, report_text)

    json_report = {
        "model_settings": {
            "kernel": SVR_KERNEL,
            "C": SVR_C,
            "gamma": SVR_GAMMA,
            "epsilon": SVR_EPSILON,
            "n_lags": N_LAGS,
            "imu_use_cols": IMU_USE_COLS,
            "train_ids": TRAIN_IDS,
            "test_ids": TEST_IDS,
            "max_train": MAX_TRAIN,
        },
        "training": {
            "x_train_shape": list(X_train.shape),
            "y_train_shape": list(y_train.shape),
            "training_time_sec": training_time_sec,
            "model_size_mb": model_size_mb,
        },
        "test_results": all_test_results,
        "single_sample_inference": single_bench,
        "batch_inference": batch_bench,
        "system_info": system_info,
        "memory_info": {
            "rss_before_train_mb": mem_before_train["rss_mb"],
            "rss_after_train_mb": mem_after_train["rss_mb"],
            "current_rss_mb": runtime_mem["rss_mb"],
            "peak_traced_memory_mb": bytes_to_mb(peak_mem_bytes),
        }
    }

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    print("\nSaved benchmark reports:")
    print(f"  {REPORT_PATH}")
    print(f"  {JSON_REPORT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()