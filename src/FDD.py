import numpy as np
import os
from scipy.signal import welch, csd, find_peaks

# =========================
# 1. Load data
# =========================

inputs = np.load("inputs.npy")   # shape: (1530, 27, 6000)
labels = np.load("labels.npy")   # shape: (1530,)

print(inputs.shape)
print(labels.shape)


# =========================
# 2. FDD: estimate PSD matrix
# =========================

def compute_psd_matrix(sample, fs=100, nperseg=1024):
    """
    sample: shape (n_sensors, n_time)
    return:
        freqs: frequency array
        psd_matrices: shape (n_freqs, n_sensors, n_sensors)
    """
    n_sensors, _ = sample.shape

    # Get frequency axis first
    freqs, _ = welch(sample[0], fs=fs, nperseg=nperseg)
    n_freqs = len(freqs)

    psd_matrices = np.zeros(
        (n_freqs, n_sensors, n_sensors),
        dtype=np.complex128
    )

    for i in range(n_sensors):
        for j in range(n_sensors):
            freqs, Pxy = csd(
                sample[i],
                sample[j],
                fs=fs,
                nperseg=nperseg
            )
            psd_matrices[:, i, j] = Pxy

    return freqs, psd_matrices


# =========================
# 3. FDD: extract natural frequencies and mode shapes
#    within predefined frequency bands
# =========================

def fdd_extract_modes(
    sample,
    fs=100,
    nperseg=1024,
    freq_bands=[(1, 5), (5, 10), (10, 20)]
):
    """
    Use Frequency Domain Decomposition:
    PSD matrix -> SVD at each frequency -> first singular value peak.

    Instead of choosing the strongest n peaks globally,
    this function searches for one peak inside each predefined frequency band.

    freq_bands:
        list of tuples, e.g.
        [(1, 5), (5, 10), (10, 20)]

    return:
        natural_freqs: shape (n_modes,)
        mode_shapes: shape (n_modes, n_sensors)
    """

    freqs, psd_matrices = compute_psd_matrix(
        sample,
        fs=fs,
        nperseg=nperseg
    )

    first_singular_values = []
    first_singular_vectors = []

    for k in range(len(freqs)):
        U, S, Vh = np.linalg.svd(psd_matrices[k])
        first_singular_values.append(S[0])
        first_singular_vectors.append(U[:, 0])

    first_singular_values = np.array(first_singular_values)
    first_singular_vectors = np.array(first_singular_vectors)

    natural_freqs = []
    mode_shapes = []

    for f_low, f_high in freq_bands:

        # Find frequency indices inside this band
        band_mask = (freqs >= f_low) & (freqs <= f_high)

        band_freqs = freqs[band_mask]
        band_singular_values = first_singular_values[band_mask]
        band_vectors = first_singular_vectors[band_mask]

        # If no frequency points in this band, use zero padding
        if len(band_freqs) == 0:
            natural_freqs.append(0.0)
            mode_shapes.append(np.zeros(sample.shape[0]))
            continue

        # Find peaks inside this band
        peaks, _ = find_peaks(band_singular_values)

        # If no clear peak, choose the maximum value in the band
        if len(peaks) == 0:
            best_idx = np.argmax(band_singular_values)
        else:
            peak_strengths = band_singular_values[peaks]
            best_peak = peaks[np.argmax(peak_strengths)]
            best_idx = best_peak

        freq = band_freqs[best_idx]
        mode_shape = band_vectors[best_idx]

        # Use real part
        mode_shape = np.real(mode_shape)

        # Normalize mode shape
        max_abs = np.max(np.abs(mode_shape))
        if max_abs > 0:
            mode_shape = mode_shape / max_abs

        natural_freqs.append(freq)
        mode_shapes.append(mode_shape)

    natural_freqs = np.array(natural_freqs)
    mode_shapes = np.array(mode_shapes)

    return natural_freqs, mode_shapes


# =========================
# 4. Compute mode shape curvature
# =========================

def compute_curvature(mode_shape, dx=1.0):
    """
    mode_shape: shape (n_sensors,)

    curvature_i = phi_{i+1} - 2phi_i + phi_{i-1} / dx^2

    return:
        curvature: shape (n_sensors - 2,)
    """
    curvature = (
        mode_shape[2:]
        - 2 * mode_shape[1:-1]
        + mode_shape[:-2]
    ) / (dx ** 2)

    return curvature


# =========================
# 5. Build feature vector
# =========================

def extract_frequency_curvature_features(
    sample,
    fs=100,
    nperseg=1024,
    freq_bands=[(1, 5), (5, 10), (10, 20)],
    dx=1.0
):
    natural_freqs, mode_shapes = fdd_extract_modes(
        sample,
        fs=fs,
        nperseg=nperseg,
        freq_bands=freq_bands
    )

    features = []

    # Add natural frequencies
    features.extend(natural_freqs)

    # Add curvature features for each mode
    for mode in mode_shapes:
        curvature = compute_curvature(mode, dx=dx)
        features.extend(curvature)

    return np.array(features)


# =========================
# 6. Extract features for all samples
# =========================

# You should adjust these bands after plotting the FDD spectrum
freq_bands = [
    (3.5, 5.5),
    (12.0, 14.5),
    (18.5, 21.5)
]

# Save extracted FDD + curvature features so we do not need to repeat
# the expensive feature extraction every time we train a model.
feature_cache_path = "X_features_frequency_curvature.npy"

if os.path.exists(feature_cache_path):
    print(f"Loading cached features from {feature_cache_path}...")
    X_features = np.load(feature_cache_path)
else:
    print("Cached features not found. Extracting FDD + curvature features...")

    X_features = []

    for idx, sample in enumerate(inputs):
        feature = extract_frequency_curvature_features(
            sample,
            fs=100,
            nperseg=1024,
            freq_bands=freq_bands,
            dx=1.0
        )

        X_features.append(feature)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(inputs)} samples")

    X_features = np.array(X_features)

    np.save(feature_cache_path, X_features)
    print(f"Saved features to {feature_cache_path}")

print("Feature shape:", X_features.shape)
