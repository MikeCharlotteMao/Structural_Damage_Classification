import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, find_peaks


# =========================
# Load data
# =========================

inputs = np.load("inputs.npy")

print("Inputs shape:", inputs.shape)


# =========================
# PSD Matrix
# =========================

def compute_psd_matrix(sample, fs=100, nperseg=512):

    n_sensors, _ = sample.shape

    freqs, _ = welch(
        sample[0],
        fs=fs,
        nperseg=nperseg
    )

    n_freqs = len(freqs)

    psd_matrices = np.zeros(
        (n_freqs, n_sensors, n_sensors),
        dtype=np.complex128
    )

    for i in range(n_sensors):
        for j in range(n_sensors):

            _, Pxy = csd(
                sample[i],
                sample[j],
                fs=fs,
                nperseg=nperseg
            )

            psd_matrices[:, i, j] = Pxy

    return freqs, psd_matrices


# =========================
# FDD
# =========================

def get_peak_frequencies(
    sample,
    fs=100,
    nperseg=512,
    min_freq=0.5,
    max_freq=30,
    prominence_ratio=0.05
):

    freqs, psd_matrices = compute_psd_matrix(
        sample,
        fs=fs,
        nperseg=nperseg
    )

    first_sv = []

    for k in range(len(freqs)):
        _, S, _ = np.linalg.svd(psd_matrices[k])
        first_sv.append(S[0])

    first_sv = np.array(first_sv)

    mask = (freqs >= min_freq) & (freqs <= max_freq)

    freqs = freqs[mask]
    first_sv = first_sv[mask]

    peaks, _ = find_peaks(
        first_sv,
        prominence=np.max(first_sv) * prominence_ratio
    )

    return freqs[peaks]


# =========================
# Scan many samples
# =========================

all_peak_freqs = []

sample_ids = range(0, 1000, 20)
# 0,20,40,...,980
# total 50 samples

for sample_id in sample_ids:

    print(f"Processing sample {sample_id}")

    peak_freqs = get_peak_frequencies(
        inputs[sample_id]
    )

    all_peak_freqs.extend(peak_freqs)


all_peak_freqs = np.array(all_peak_freqs)

print("\nTotal peaks found:", len(all_peak_freqs))


# =========================
# Histogram
# =========================

plt.figure(figsize=(14, 7))

hist, edges, _ = plt.hist(
    all_peak_freqs,
    bins=80,
    alpha=0.8
)

plt.xlabel("Peak Frequency (Hz)")
plt.ylabel("Count")
plt.title(
    "Peak Frequency Distribution Across Samples"
)

plt.grid(True)


# =========================
# Find top frequency regions
# =========================

top_bins = np.argsort(hist)[-10:]

for idx in top_bins:

    center = (
        edges[idx]
        + edges[idx+1]
    ) / 2

    count = int(hist[idx])

    # vertical dashed line
    plt.axvline(
        center,
        linestyle="--",
        alpha=0.6
    )

    # label
    plt.text(
        center,
        count + 1,
        f"{center:.2f} Hz\n({count})",
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()


# =========================
# Print most common regions
# =========================

hist, edges = np.histogram(
    all_peak_freqs,
    bins=80
)

top_bins = np.argsort(hist)[-10:]

print("\nMost populated frequency regions:")

for idx in reversed(top_bins):

    center = (
        edges[idx]
        + edges[idx+1]
    ) / 2

    print(
        f"{center:.2f} Hz "
        f"-> {hist[idx]} peaks"
    )
