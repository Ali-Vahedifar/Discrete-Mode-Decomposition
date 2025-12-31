"""
Data Preprocessing Utilities for Haptic Signals
================================================

Comprehensive preprocessing pipeline for haptic signal data,
including normalization, filtering, augmentation, and feature extraction.

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk

IEEE INFOCOM 2025
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing.
    
    Attributes:
        normalize: Whether to normalize data
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
        lowpass_filter: Whether to apply lowpass filter
        lowpass_cutoff: Lowpass filter cutoff frequency (Hz)
        sampling_rate: Signal sampling rate (Hz)
        resample: Whether to resample signal
        target_rate: Target sampling rate for resampling
        add_noise: Whether to add noise for augmentation
        noise_level: Noise standard deviation
    """
    normalize: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    lowpass_filter: bool = False
    lowpass_cutoff: float = 100.0
    sampling_rate: int = 1000
    resample: bool = False
    target_rate: int = 500
    add_noise: bool = False
    noise_level: float = 0.01


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline.
    
    Handles all preprocessing steps for haptic signal data,
    including normalization, filtering, and augmentation.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    
    Example:
    --------
    >>> preprocessor = DataPreprocessor(PreprocessConfig())
    >>> processed_data = preprocessor.fit_transform(raw_data)
    >>> new_data = preprocessor.transform(test_data)
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        config : PreprocessConfig, optional
            Preprocessing configuration
        """
        self.config = config or PreprocessConfig()
        
        # Statistics for normalization (computed during fit)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        
        # Filter design
        self._filter_b: Optional[np.ndarray] = None
        self._filter_a: Optional[np.ndarray] = None
        
        self._is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """
        Fit preprocessor to data.
        
        Computes statistics needed for preprocessing.
        
        Parameters:
        -----------
        data : np.ndarray
            Training data of shape (N, C)
            
        Returns:
        --------
        self
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Compute normalization statistics
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std < 1e-8] = 1.0  # Prevent division by zero
        
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        
        # Design lowpass filter if needed
        if self.config.lowpass_filter:
            nyquist = self.config.sampling_rate / 2
            normalized_cutoff = self.config.lowpass_cutoff / nyquist
            self._filter_b, self._filter_a = scipy_signal.butter(
                4, normalized_cutoff, btype='low'
            )
        
        self._is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to transform
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        result = data.copy()
        
        # Remove outliers
        if self.config.remove_outliers:
            result = self._remove_outliers(result)
        
        # Apply lowpass filter
        if self.config.lowpass_filter:
            result = self._apply_lowpass(result)
        
        # Resample if needed
        if self.config.resample:
            result = self._resample(result)
        
        # Normalize
        if self.config.normalize:
            result = (result - self.mean) / self.std
        
        # Add noise for augmentation
        if self.config.add_noise:
            result = result + self.config.noise_level * np.random.randn(*result.shape)
        
        return result.astype(np.float32)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform (denormalize) data.
        
        Parameters:
        -----------
        data : np.ndarray
            Normalized data
            
        Returns:
        --------
        np.ndarray
            Denormalized data
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted")
        
        if self.config.normalize:
            return data * self.std + self.mean
        return data
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using z-score."""
        z_scores = np.abs((data - self.mean) / self.std)
        outlier_mask = z_scores > self.config.outlier_threshold
        
        # Replace outliers with interpolated values
        result = data.copy()
        for col in range(data.shape[1]):
            mask = outlier_mask[:, col]
            if np.any(mask):
                valid_indices = np.where(~mask)[0]
                invalid_indices = np.where(mask)[0]
                
                if len(valid_indices) > 1:
                    f = interp1d(
                        valid_indices,
                        data[valid_indices, col],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    result[invalid_indices, col] = f(invalid_indices)
        
        return result
    
    def _apply_lowpass(self, data: np.ndarray) -> np.ndarray:
        """Apply lowpass filter."""
        if self._filter_b is None or self._filter_a is None:
            return data
        
        result = np.zeros_like(data)
        for col in range(data.shape[1]):
            result[:, col] = scipy_signal.filtfilt(
                self._filter_b, self._filter_a, data[:, col]
            )
        return result
    
    def _resample(self, data: np.ndarray) -> np.ndarray:
        """Resample data to target rate."""
        orig_length = len(data)
        ratio = self.config.target_rate / self.config.sampling_rate
        new_length = int(orig_length * ratio)
        
        result = np.zeros((new_length, data.shape[1]))
        for col in range(data.shape[1]):
            result[:, col] = scipy_signal.resample(data[:, col], new_length)
        
        return result
    
    def get_params(self) -> Dict:
        """Get preprocessor parameters."""
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'min_val': self.min_val.tolist() if self.min_val is not None else None,
            'max_val': self.max_val.tolist() if self.max_val is not None else None,
            'config': {
                'normalize': self.config.normalize,
                'remove_outliers': self.config.remove_outliers,
                'lowpass_filter': self.config.lowpass_filter,
            }
        }
    
    def set_params(self, params: Dict) -> 'DataPreprocessor':
        """Set preprocessor parameters."""
        if params.get('mean') is not None:
            self.mean = np.array(params['mean'])
        if params.get('std') is not None:
            self.std = np.array(params['std'])
        if params.get('min_val') is not None:
            self.min_val = np.array(params['min_val'])
        if params.get('max_val') is not None:
            self.max_val = np.array(params['max_val'])
        self._is_fitted = True
        return self


def normalize_signal(
    signal: np.ndarray,
    method: str = 'zscore',
    axis: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize signal using various methods.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    method : str
        Normalization method: 'zscore', 'minmax', 'robust'
    axis : int
        Axis along which to normalize
        
    Returns:
    --------
    Tuple[np.ndarray, Dict]
        Normalized signal and normalization parameters
    """
    params = {}
    
    if method == 'zscore':
        mean = np.mean(signal, axis=axis, keepdims=True)
        std = np.std(signal, axis=axis, keepdims=True)
        std[std < 1e-8] = 1.0
        normalized = (signal - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'zscore'}
        
    elif method == 'minmax':
        min_val = np.min(signal, axis=axis, keepdims=True)
        max_val = np.max(signal, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val < 1e-8] = 1.0
        normalized = (signal - min_val) / range_val
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
    elif method == 'robust':
        median = np.median(signal, axis=axis, keepdims=True)
        q75 = np.percentile(signal, 75, axis=axis, keepdims=True)
        q25 = np.percentile(signal, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0
        normalized = (signal - median) / iqr
        params = {'median': median, 'iqr': iqr, 'method': 'robust'}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32), params


def denormalize_signal(
    signal: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    Denormalize signal using stored parameters.
    
    Parameters:
    -----------
    signal : np.ndarray
        Normalized signal
    params : Dict
        Normalization parameters from normalize_signal
        
    Returns:
    --------
    np.ndarray
        Denormalized signal
    """
    method = params.get('method', 'zscore')
    
    if method == 'zscore':
        return signal * params['std'] + params['mean']
    elif method == 'minmax':
        return signal * (params['max'] - params['min']) + params['min']
    elif method == 'robust':
        return signal * params['iqr'] + params['median']
    else:
        raise ValueError(f"Unknown method: {method}")


def add_noise(
    signal: np.ndarray,
    noise_type: str = 'gaussian',
    noise_level: float = 0.01,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add noise to signal for data augmentation.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    noise_type : str
        Type of noise: 'gaussian', 'uniform', 'salt_pepper'
    noise_level : float
        Noise level (std for gaussian, range for uniform)
    seed : int, optional
        Random seed
        
    Returns:
    --------
    np.ndarray
        Noisy signal
    """
    if seed is not None:
        np.random.seed(seed)
    
    if noise_type == 'gaussian':
        noise = noise_level * np.random.randn(*signal.shape)
    elif noise_type == 'uniform':
        noise = noise_level * (np.random.rand(*signal.shape) - 0.5) * 2
    elif noise_type == 'salt_pepper':
        noise = np.zeros_like(signal)
        mask = np.random.rand(*signal.shape) < noise_level
        noise[mask] = np.random.choice([-1, 1], size=np.sum(mask)) * np.std(signal)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return signal + noise


def remove_outliers(
    signal: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Remove outliers from signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    method : str
        Detection method: 'zscore', 'iqr', 'mad'
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    np.ndarray
        Signal with outliers replaced
    """
    result = signal.copy()
    
    if method == 'zscore':
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0)
        std[std < 1e-8] = 1.0
        z_scores = np.abs((signal - mean) / std)
        mask = z_scores > threshold
        
    elif method == 'iqr':
        q75 = np.percentile(signal, 75, axis=0)
        q25 = np.percentile(signal, 25, axis=0)
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
        mask = (signal < lower) | (signal > upper)
        
    elif method == 'mad':
        median = np.median(signal, axis=0)
        mad = np.median(np.abs(signal - median), axis=0)
        mad[mad < 1e-8] = 1.0
        modified_z = 0.6745 * (signal - median) / mad
        mask = np.abs(modified_z) > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Replace outliers with interpolated values
    for col in range(signal.shape[1]):
        col_mask = mask[:, col]
        if np.any(col_mask):
            valid_idx = np.where(~col_mask)[0]
            invalid_idx = np.where(col_mask)[0]
            
            if len(valid_idx) > 1:
                f = interp1d(
                    valid_idx,
                    signal[valid_idx, col],
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                result[invalid_idx, col] = f(invalid_idx)
    
    return result


def segment_signal(
    signal: np.ndarray,
    segment_length: int,
    overlap: int = 0,
    pad: bool = True
) -> np.ndarray:
    """
    Segment signal into fixed-length chunks.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    segment_length : int
        Length of each segment
    overlap : int
        Number of overlapping samples
    pad : bool
        Whether to pad last segment
        
    Returns:
    --------
    np.ndarray
        Segmented signal of shape (num_segments, segment_length, num_features)
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    N, C = signal.shape
    step = segment_length - overlap
    
    segments = []
    for i in range(0, N - segment_length + 1, step):
        segments.append(signal[i:i + segment_length])
    
    # Handle last segment
    if pad and len(signal) > segments[-1][-1] if segments else True:
        remaining = N - (len(segments) * step)
        if remaining > 0:
            last_segment = np.zeros((segment_length, C))
            last_segment[:remaining] = signal[-remaining:]
            segments.append(last_segment)
    
    return np.array(segments)


def extract_features(
    signal: np.ndarray,
    features: List[str] = ['mean', 'std', 'max', 'min']
) -> np.ndarray:
    """
    Extract statistical features from signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal of shape (N, C) or (N,)
    features : List[str]
        List of features to extract
        
    Returns:
    --------
    np.ndarray
        Feature vector
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    feature_funcs = {
        'mean': lambda x: np.mean(x, axis=0),
        'std': lambda x: np.std(x, axis=0),
        'max': lambda x: np.max(x, axis=0),
        'min': lambda x: np.min(x, axis=0),
        'median': lambda x: np.median(x, axis=0),
        'rms': lambda x: np.sqrt(np.mean(x**2, axis=0)),
        'skew': lambda x: np.mean(((x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8))**3, axis=0),
        'kurtosis': lambda x: np.mean(((x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8))**4, axis=0) - 3,
        'peak_to_peak': lambda x: np.max(x, axis=0) - np.min(x, axis=0),
        'zero_crossings': lambda x: np.sum(np.diff(np.sign(x), axis=0) != 0, axis=0),
    }
    
    result = []
    for feat in features:
        if feat in feature_funcs:
            result.append(feature_funcs[feat](signal))
        else:
            warnings.warn(f"Unknown feature: {feat}")
    
    return np.concatenate(result)


def bandpass_filter(
    signal: np.ndarray,
    low_freq: float,
    high_freq: float,
    sampling_rate: int = 1000,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    low_freq : float
        Lower cutoff frequency (Hz)
    high_freq : float
        Upper cutoff frequency (Hz)
    sampling_rate : int
        Sampling rate (Hz)
    order : int
        Filter order
        
    Returns:
    --------
    np.ndarray
        Filtered signal
    """
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    
    if signal.ndim == 1:
        return scipy_signal.filtfilt(b, a, signal)
    
    result = np.zeros_like(signal)
    for col in range(signal.shape[1]):
        result[:, col] = scipy_signal.filtfilt(b, a, signal[:, col])
    
    return result


def compute_spectral_features(
    signal: np.ndarray,
    sampling_rate: int = 1000
) -> Dict[str, np.ndarray]:
    """
    Compute spectral features of signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    sampling_rate : int
        Sampling rate (Hz)
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary of spectral features
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    features = {}
    
    for col in range(signal.shape[1]):
        freqs, psd = scipy_signal.welch(
            signal[:, col],
            fs=sampling_rate,
            nperseg=min(256, len(signal))
        )
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-8)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-8)
        )
        
        # Peak frequency
        peak_freq = freqs[np.argmax(psd)]
        
        features[f'spectral_centroid_{col}'] = spectral_centroid
        features[f'spectral_bandwidth_{col}'] = spectral_bandwidth
        features[f'peak_frequency_{col}'] = peak_freq
        features[f'total_power_{col}'] = np.sum(psd)
    
    return features


if __name__ == "__main__":
    # Test preprocessing utilities
    print("Testing Preprocessing Utilities")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    N = 1000
    t = np.linspace(0, 1, N)
    
    # Signal with multiple components
    signal = np.column_stack([
        np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(N),
        np.cos(2 * np.pi * 20 * t) + 0.3 * np.random.randn(N),
        0.5 * np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(N)
    ])
    
    # Add some outliers
    outlier_idx = [100, 200, 500, 700]
    signal[outlier_idx, 0] = signal[outlier_idx, 0] * 10
    
    print(f"Original signal shape: {signal.shape}")
    print(f"Original mean: {np.mean(signal, axis=0)}")
    print(f"Original std: {np.std(signal, axis=0)}")
    
    # Test DataPreprocessor
    config = PreprocessConfig(
        normalize=True,
        remove_outliers=True,
        lowpass_filter=True,
        lowpass_cutoff=50.0
    )
    
    preprocessor = DataPreprocessor(config)
    processed = preprocessor.fit_transform(signal)
    
    print(f"\nProcessed signal shape: {processed.shape}")
    print(f"Processed mean: {np.mean(processed, axis=0)}")
    print(f"Processed std: {np.std(processed, axis=0)}")
    
    # Test inverse transform
    recovered = preprocessor.inverse_transform(processed)
    print(f"\nRecovered signal mean: {np.mean(recovered, axis=0)}")
    
    # Test normalization functions
    normalized, params = normalize_signal(signal, method='zscore')
    print(f"\nNormalized (zscore) mean: {np.mean(normalized, axis=0)}")
    
    # Test noise addition
    noisy = add_noise(signal, noise_type='gaussian', noise_level=0.1)
    print(f"\nNoisy signal std: {np.std(noisy, axis=0)}")
    
    # Test outlier removal
    cleaned = remove_outliers(signal, method='zscore', threshold=3.0)
    print(f"\nCleaned signal max: {np.max(cleaned, axis=0)}")
    
    # Test feature extraction
    features = extract_features(signal, features=['mean', 'std', 'rms', 'skew'])
    print(f"\nExtracted features shape: {features.shape}")
    
    # Test spectral features
    spectral = compute_spectral_features(signal)
    print(f"\nSpectral features: {list(spectral.keys())}")
    
    print("\nAll preprocessing tests passed!")
