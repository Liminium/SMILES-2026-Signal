import json
import os
import numpy as np
from scipy.io import loadmat
from task_and_baseline import baseline, build_task_helpers, MODEL_SUBSET

# Path to the file
data_path = "challenge_challenge.mat"

data = loadmat(data_path, simplify_cells=True)

tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx, win_size, overlap):
    tx_nonlinear_pred = helpers["fit_tx_prediction"](rx)

    residual = rx - tx_nonlinear_pred

    spatial_cancellation = np.zeros_like(rx)

    #  80000, 47000 - 10   71000, 36000 - 10.10

    weight = np.zeros((N, 1))

    for start in range(0, N, win_size - overlap):

        end = min(start + win_size, N)

        chunk = residual[start:end]

        chunk_filtered = np.column_stack([
            helpers["score_filter"](chunk[:, ch])
            for ch in range(rx.shape[1])
        ])

        analysis_window = chunk_filtered

        U, s, vh = np.linalg.svd(
            analysis_window,
            full_matrices=False
        )

        v1 = vh[0, :].conj()
        v2 = vh[1, :].conj()

        shared_signal_1 = chunk_filtered @ v1
        shared_signal_2 = chunk_filtered @ v2

        local_cancel = np.zeros_like(chunk)

        for ch in range(rx.shape[1]):
            alpha1 = np.vdot(shared_signal_1, chunk_filtered[:, ch]) / (np.vdot(shared_signal_1, shared_signal_1) + 1e-30)
            coherent1 = alpha1 * shared_signal_1
            beta1 = np.vdot(coherent1, chunk[:, ch]) (np.vdot(coherent1, coherent1) + 1e-30)

            alpha2 = np.vdot(shared_signal_2, chunk_filtered[:, ch]) / (np.vdot(shared_signal_2, shared_signal_2) + 1e-30)
            coherent2 = alpha2 * shared_signal_2
            beta2 = np.vdot(coherent2, chunk[:, ch]) / (np.vdot(coherent2, coherent2) + 1e-30)

            d = s[0] / (s[1] + 1e-30)

            k1 = 0.58 + 0.12 * np.tanh((d - 4) / 3)  # Pure heuristics that gave additional 0.09 performance
            k2 = 0.08
            if d < 2:
                k2 = 0.12

            local_cancel[:, ch] = k1 * beta1 * coherent1 + k2 * beta2 * coherent2

        spatial_cancellation[start:end] += local_cancel
        weight[start:end] += 1.0

    spatial_cancellation /= np.maximum(weight, 1e-30)

    filtered_before = np.column_stack([
        helpers["score_filter"](residual[:, ch]) for ch in range(4)])

    filtered_after = np.column_stack([
        helpers["score_filter"](
            residual[:, ch] - spatial_cancellation[:, ch]
        )
        for ch in range(4)
    ])

    p_before = np.mean(np.abs(filtered_before) ** 2)
    p_after = np.mean(np.abs(filtered_after) ** 2)

    if p_after > 0.78 * p_before:
        spatial_cancellation *= 0.9

    return residual - spatial_cancellation


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
