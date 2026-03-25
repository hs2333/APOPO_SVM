# 6-classification_svm_benchmark.py
# Capstone: 6-state motion classification with SVM (RBF)
# Added benchmarking for:
# - model accuracy / confusion matrix / classification report
# - inference latency
# - throughput (inferences per second)
# - hardware / system information
# - process memory usage
# - model file size
# - saved benchmark report

import os
import time
import json
import pickle
import platform
import statistics
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt
import psutil
import sklearn

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

MODEL_PATH = "svm_rbf_model.pkl"
REPORT_PATH = "svm_benchmark_report.txt"
JSON_REPORT_PATH = "svm_benchmark_report.json"


# -----------------------------
# Benchmark settings
# -----------------------------
WARMUP_RUNS = 100
TIMED_RUNS_SINGLE = 1000
TIMED_RUNS_BATCH = 300


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


def bytes_to_mb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 2)


def bytes_to_gb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


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
        "rss_mb": bytes_to_mb(mem.rss),   # resident set size
        "vms_mb": bytes_to_mb(mem.vms),   # virtual memory size
    }


def benchmark_single_sample_latency(model, x_single, warmup_runs=100, timed_runs=1000):
    """
    Measure single-sample latency.
    x_single shape should be (1, n_features).
    """
    # Warmup
    for _ in range(warmup_runs):
        model.predict(x_single)

    latencies = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        model.predict(x_single)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    avg_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    p95_ms = float(np.percentile(latencies, 95))

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


def benchmark_batch_throughput(model, X_batch, warmup_runs=100, timed_runs=300):
    """
    Measure batch throughput using the whole test batch.
    This reports:
    - batches/sec
    - samples/sec
    """
    n_samples = len(X_batch)

    # Warmup
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
N_test = len(X_test)
print(f"\nN_train = {N_train}")
print(f"N_test  = {N_test}")


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

# Track peak Python allocations during training section
tracemalloc.start()
mem_before_train = get_process_memory_info()

train_t0 = time.perf_counter()
svm.fit(X_train, y_train)
train_t1 = time.perf_counter()

mem_after_train = get_process_memory_info()
current_mem_bytes, peak_mem_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()

training_time_sec = train_t1 - train_t0

pred = svm.predict(X_test)
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=labels)
report_str = classification_report(y_test, pred, labels=labels, target_names=target_names)

print("\nSVM (RBF) Holdout (10%) results:")
print(f"  C={C}, gamma={gamma}")
print(f"  Accuracy: {acc:.4f}")
print(f"  Training time: {training_time_sec:.4f} s")

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(report_str)


# -----------------------------
# Save model + report model size
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "model": svm,
            "scaler": scaler,
            "feature_set": FEATURE_SET,
            "label_names": label_names,
            "C": C,
            "gamma": gamma,
        },
        f
    )

model_size_mb = bytes_to_mb(os.path.getsize(MODEL_PATH))
print(f"\nSaved model: {MODEL_PATH}")
print(f"Model file size: {model_size_mb:.4f} MB")


# -----------------------------
# Benchmark inference
# -----------------------------
# Single-sample latency benchmark
x_single = X_test[0:1]

single_bench = benchmark_single_sample_latency(
    svm,
    x_single,
    warmup_runs=WARMUP_RUNS,
    timed_runs=TIMED_RUNS_SINGLE
)

# Batch throughput benchmark
batch_bench = benchmark_batch_throughput(
    svm,
    X_test,
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


# -----------------------------
# System / memory info
# -----------------------------
system_info = get_system_info()
runtime_mem = get_process_memory_info()

print("\nSystem / hardware info:")
for k, v in system_info.items():
    print(f"  {k}: {v}")

print("\nProcess memory usage:")
print(f"  Before training RSS : {mem_before_train['rss_mb']:.3f} MB")
print(f"  After training RSS  : {mem_after_train['rss_mb']:.3f} MB")
print(f"  Current process RSS : {runtime_mem['rss_mb']:.3f} MB")
print(f"  Peak traced memory  : {bytes_to_mb(peak_mem_bytes):.3f} MB")


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
# The SVM trained above is in 11D. For a valid 2D decision-region plot, train a NEW SVM in PCA space.
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


# -----------------------------
# Save benchmark report
# -----------------------------
report_lines = []
report_lines.append("SVM Benchmark Report")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append("Model settings:")
report_lines.append(f"  Kernel: rbf")
report_lines.append(f"  C: {C}")
report_lines.append(f"  gamma: {gamma}")
report_lines.append(f"  Num train samples: {N_train}")
report_lines.append(f"  Num test samples: {N_test}")
report_lines.append("")

report_lines.append("Selected features:")
for f in FEATURE_SET:
    report_lines.append(f"  - {f}")
report_lines.append("")

report_lines.append("Classification performance:")
report_lines.append(f"  Accuracy: {acc:.6f}")
report_lines.append(f"  Training time (s): {training_time_sec:.6f}")
report_lines.append("")
report_lines.append("Confusion matrix:")
report_lines.append(str(cm))
report_lines.append("")
report_lines.append("Classification report:")
report_lines.append(report_str)
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

report_lines.append("Memory / storage:")
report_lines.append(f"  Process RSS before training (MB): {mem_before_train['rss_mb']:.6f}")
report_lines.append(f"  Process RSS after training (MB): {mem_after_train['rss_mb']:.6f}")
report_lines.append(f"  Current process RSS (MB): {runtime_mem['rss_mb']:.6f}")
report_lines.append(f"  Peak traced memory (MB): {bytes_to_mb(peak_mem_bytes):.6f}")
report_lines.append(f"  Saved model size (MB): {model_size_mb:.6f}")
report_lines.append("")

report_text = "\n".join(report_lines)
save_text_report(REPORT_PATH, report_text)

json_report = {
    "model_settings": {
        "kernel": "rbf",
        "C": C,
        "gamma": gamma,
        "num_train_samples": N_train,
        "num_test_samples": N_test,
        "feature_set": FEATURE_SET,
    },
    "classification_performance": {
        "accuracy": acc,
        "training_time_sec": training_time_sec,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_str,
    },
    "single_sample_inference": single_bench,
    "batch_inference": batch_bench,
    "system_info": system_info,
    "memory_info": {
        "rss_before_train_mb": mem_before_train["rss_mb"],
        "rss_after_train_mb": mem_after_train["rss_mb"],
        "current_rss_mb": runtime_mem["rss_mb"],
        "peak_traced_memory_mb": bytes_to_mb(peak_mem_bytes),
        "model_size_mb": model_size_mb,
    },
}

with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(json_report, f, indent=2)

print("\nSaved benchmark reports:")
print(f"  {REPORT_PATH}")
print(f"  {JSON_REPORT_PATH}")