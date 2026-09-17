"""Feature extraction from EEG signals: band power via FFT and trial segmentation."""

import numpy as np
from scipy.fft import fft

from bci.config import ALPHA_BAND_HZ, BETA_BAND_HZ, SAMPLING_RATE_HZ


def band_power_spectral_density(signal, low_hz, high_hz, sampling_rate_hz=SAMPLING_RATE_HZ):
    """One-sided power spectral density of `signal`, restricted to [low_hz, high_hz]."""
    n_samples = len(signal)
    spectrum = fft(signal)
    power = np.abs(spectrum / n_samples) ** 2
    power = power[: n_samples // 2 + 1]
    power[1:-1] = 2 * power[1:-1]
    frequency_resolution = sampling_rate_hz / n_samples
    power_density = power / frequency_resolution
    frequencies = sampling_rate_hz * np.arange(0, n_samples // 2 + 1) / n_samples
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return power_density[band_mask], frequencies[band_mask]


def alpha_beta_band_power(channel_segment):
    """Mean alpha- and beta-band power for one channel segment.

    Shared by training-time feature extraction and the live classifier so both
    use the exact same band definitions.
    """
    alpha_power = np.mean(band_power_spectral_density(channel_segment, *ALPHA_BAND_HZ)[0])
    beta_power = np.mean(band_power_spectral_density(channel_segment, *BETA_BAND_HZ)[0])
    return alpha_power, beta_power


def segment_trials_and_extract_features(labels, channel_c3, channel_c4):
    """Split a continuous recording into trials and extract alpha/beta band-power features.

    A trial's active period runs from where `labels` rises from the rest class (0)
    to an active class (1 or 2), up to where it falls back down to rest.
    """
    active_starts = []
    trial_ends = []

    for i in range(len(labels) - 1):
        if labels[i] < labels[i + 1]:
            active_starts.append(i + 1)
        if labels[i] > labels[i + 1]:
            trial_ends.append(i)
    trial_ends.append(len(labels))

    features, trial_labels = [], []
    for start, end in zip(active_starts, trial_ends):
        end += 1
        c3_segment = channel_c3[start:end]
        c4_segment = channel_c4[start:end]
        label_segment = labels[start:end]

        c3_alpha, c3_beta = alpha_beta_band_power(c3_segment)
        c4_alpha, c4_beta = alpha_beta_band_power(c4_segment)

        features.append([c3_alpha, c3_beta, c4_alpha, c4_beta])
        trial_labels.append(np.unique(label_segment)[0])

    return np.array(features), np.array(trial_labels)
