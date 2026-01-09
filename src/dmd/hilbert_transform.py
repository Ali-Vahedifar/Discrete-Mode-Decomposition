"""
Discrete Hilbert Transform Implementation
=========================================

Author: Ali Vahedi
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
IEEE INFOCOM 2025
This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

Implementation of the Discrete Hilbert Transform for computing analytic signals
as part of the DMD algorithm.

Mathematical Background (Equations 6-8):
---------------------------------------
The N-point Hilbert transform is characterized by its DFT coefficients:

    H[k] = { -j  for 1 ≤ k < N/2
           {  0  for k = 0, N/2
           {  j  for N/2 < k ≤ N-1

The impulse response is (Eq. 7):
    h[n] = (2 * sin²(πn/2)) / (πn) = (1 - (-1)^n) / (πn),  n ≠ 0
    h[0] = 0

The analytic signal q[n] associated with a real discrete signal x[n] is (Eq. 8):
    q[n] = x[n] + j * H{x[n]}

This allows extraction of:
- Instantaneous amplitude: A[n] = |q[n]|
- Instantaneous phase: φ[n] = arg(q[n])
- Instantaneous frequency: ω[n] = φ[n+1] - φ[n]
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Tuple, Optional, Union
from scipy.signal import hilbert as scipy_hilbert


class DiscreteHilbertTransform:
    """
    Discrete Hilbert Transform for computing analytic signals.
    
    This class implements the discrete Hilbert transform as defined in the paper,
    used for computing instantaneous amplitude and frequency of IMFs.
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    
    The Hilbert transform converts a real signal into its analytic signal,
    which has only positive frequency components. This is essential for
    computing instantaneous amplitude and phase.
    
    Attributes:
        use_scipy: Whether to use scipy's implementation as backend
        
    Example:
    --------
    >>> hilbert = DiscreteHilbertTransform()
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> analytic = hilbert.transform(signal)
    >>> amplitude = hilbert.get_envelope(signal)
    >>> phase = hilbert.get_phase(signal)
    """
    
    def __init__(self, use_scipy: bool = False):
        """
        Initialize the Hilbert transform.
        
        Parameters:
        -----------
        use_scipy : bool
            If True, use scipy.signal.hilbert as backend.
            If False, use custom implementation based on paper equations.
            Default is False.
        """
        self.use_scipy = use_scipy
        self._cached_H = {}  # Cache transfer functions for different lengths
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the Hilbert transform of a signal.
        
        This returns the imaginary part of the analytic signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input real signal of shape (N,) or (N, C)
            
        Returns:
        --------
        np.ndarray
            Hilbert transform of the signal (imaginary part of analytic signal)
        """
        if signal.ndim == 1:
            return self._transform_1d(signal)
        elif signal.ndim == 2:
            transformed = np.zeros_like(signal)
            for c in range(signal.shape[1]):
                transformed[:, c] = self._transform_1d(signal[:, c])
            return transformed
        else:
            raise ValueError(f"Signal must be 1D or 2D, got shape {signal.shape}")
    
    def _transform_1d(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Hilbert transform of a 1D signal.
        
        Implements Equations 6-8 from the paper.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal of shape (N,)
            
        Returns:
        --------
        np.ndarray
            Hilbert transform
        """
        if self.use_scipy:
            analytic = scipy_hilbert(signal)
            return np.imag(analytic)
        
        N = len(signal)
        
        # Get or compute transfer function - Eq. 6
        H = self._get_transfer_function(N)
        
        # Compute DFT
        X = fft(signal)
        
        # Apply Hilbert transfer function
        Y = X * H
        
        # Return to time domain
        return np.real(ifft(Y))
    
    def _get_transfer_function(self, N: int) -> np.ndarray:
        """
        Get the Hilbert transform transfer function.
        
        Implements Equation 6 from the paper:
            H[k] = { -j  for 1 ≤ k < N/2
                   {  0  for k = 0, N/2
                   {  j  for N/2 < k ≤ N-1
        
        Parameters:
        -----------
        N : int
            Signal length
            
        Returns:
        --------
        np.ndarray
            Transfer function coefficients
        """
        if N in self._cached_H:
            return self._cached_H[N]
        
        H = np.zeros(N, dtype=complex)
        
        # Positive frequencies: -j (multiply by -j to get Hilbert transform)
        H[1:N//2] = -1j
        
        # Negative frequencies: +j
        H[N//2+1:] = 1j
        
        # DC and Nyquist: 0
        H[0] = 0
        if N % 2 == 0:
            H[N//2] = 0
        
        self._cached_H[N] = H
        return H
    
    def get_analytic_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the analytic signal.
        
        Implements Equation 8 from the paper:
            z[n] = x[n] + j * H{x[n]}
        
        Parameters:
        -----------
        signal : np.ndarray
            Input real signal
            
        Returns:
        --------
        np.ndarray
            Complex analytic signal
        """
        if self.use_scipy:
            return scipy_hilbert(signal)
        
        hilbert_transform = self.transform(signal)
        return signal + 1j * hilbert_transform
    
    def get_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the instantaneous amplitude (envelope) of a signal.
        
        A[n] = |z[n]| = sqrt(x[n]² + H{x[n]}²)
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        np.ndarray
            Instantaneous amplitude envelope
        """
        analytic = self.get_analytic_signal(signal)
        return np.abs(analytic)
    
    def get_phase(self, signal: np.ndarray, unwrap: bool = True) -> np.ndarray:
        """
        Compute the instantaneous phase of a signal.
        
        φ[n] = arg(z[n]) = atan2(H{x[n]}, x[n])
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        unwrap : bool
            If True, unwrap the phase to avoid discontinuities.
            Default is True.
            
        Returns:
        --------
        np.ndarray
            Instantaneous phase
        """
        analytic = self.get_analytic_signal(signal)
        phase = np.angle(analytic)
        
        if unwrap:
            phase = np.unwrap(phase)
        
        return phase
    
    def get_instantaneous_frequency(
        self,
        signal: np.ndarray,
        fs: float = 1.0
    ) -> np.ndarray:
        """
        Compute the instantaneous frequency of a signal.
        
        ω[n] = (φ[n+1] - φ[n]) / (2π)
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        fs : float
            Sampling frequency. Default is 1.0 (normalized).
            
        Returns:
        --------
        np.ndarray
            Instantaneous frequency
        """
        phase = self.get_phase(signal, unwrap=True)
        
        # Compute phase derivative
        inst_freq = np.diff(phase) * fs / (2 * np.pi)
        
        # Pad to maintain length
        inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
        
        return inst_freq
    
    def get_impulse_response(self, N: int) -> np.ndarray:
        """
        Get the impulse response of the Hilbert transform.
        
        Implements Equation 7 from the paper:
            h[n] = (1 - (-1)^n) / (πn),  n ≠ 0
            h[0] = 0
        
        Parameters:
        -----------
        N : int
            Length of impulse response
            
        Returns:
        --------
        np.ndarray
            Impulse response
        """
        n = np.arange(N) - N // 2  # Center around 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            h = (1 - (-1) ** n) / (np.pi * n)
        
        # Handle n = 0
        h[N // 2] = 0
        
        return h
    
    def decompose_am_fm(
        self,
        signal: np.ndarray,
        fs: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose signal into amplitude and frequency modulation components.
        
        A signal can be represented as:
            x[n] = A[n] * cos(φ[n])
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        fs : float
            Sampling frequency
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Amplitude A[n], phase φ[n], and instantaneous frequency ω[n]
        """
        amplitude = self.get_envelope(signal)
        phase = self.get_phase(signal, unwrap=True)
        inst_freq = self.get_instantaneous_frequency(signal, fs)
        
        return amplitude, phase, inst_freq
    
    def is_imf(
        self,
        signal: np.ndarray,
        tolerance: float = 0.1
    ) -> Tuple[bool, dict]:
        """
        Check if a signal is a valid Intrinsic Mode Function (IMF).
        
        An IMF should have:
        1. Number of extrema and zero crossings differ by at most 1
        2. Mean of upper and lower envelopes is approximately zero
        3. Instantaneous frequency is non-negative
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal to check
        tolerance : float
            Tolerance for envelope mean check
            
        Returns:
        --------
        Tuple[bool, dict]
            Boolean indicating if signal is IMF, and dictionary with details
        """
        # Count extrema
        diff_signal = np.diff(signal)
        sign_changes = np.diff(np.sign(diff_signal))
        num_extrema = np.sum(np.abs(sign_changes) > 0)
        
        # Count zero crossings
        sign_signal = np.sign(signal)
        zero_crossings = np.sum(np.abs(np.diff(sign_signal)) > 0)
        
        # Check extrema/zero-crossing condition
        extrema_condition = np.abs(num_extrema - zero_crossings) <= 1
        
        # Compute envelope and check mean
        amplitude = self.get_envelope(signal)
        envelope_mean = np.mean(amplitude * np.sign(signal))
        envelope_condition = np.abs(envelope_mean) < tolerance * np.max(np.abs(signal))
        
        # Check instantaneous frequency
        inst_freq = self.get_instantaneous_frequency(signal)
        freq_condition = np.mean(inst_freq >= -tolerance) > 0.95
        
        is_valid = extrema_condition and envelope_condition and freq_condition
        
        details = {
            'num_extrema': num_extrema,
            'num_zero_crossings': zero_crossings,
            'extrema_condition': extrema_condition,
            'envelope_mean': envelope_mean,
            'envelope_condition': envelope_condition,
            'freq_condition': freq_condition,
            'mean_inst_freq': np.mean(inst_freq)
        }
        
        return is_valid, details


class HilbertHuangTransform:
    """
    Hilbert-Huang Transform (HHT) for time-frequency analysis.
    
    Combines empirical mode decomposition with Hilbert spectral analysis.
    
    Author: Ali Vahedi
    Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
    IEEE INFOCOM 2025
    """
    
    def __init__(self):
        """Initialize HHT."""
        self.hilbert = DiscreteHilbertTransform()
    
    def compute_hilbert_spectrum(
        self,
        modes: np.ndarray,
        fs: float = 1.0,
        num_freq_bins: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Hilbert spectrum from modes.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of modes, shape (M, N)
        fs : float
            Sampling frequency
        num_freq_bins : int
            Number of frequency bins
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Time axis, frequency axis, and spectrum matrix
        """
        if modes.ndim == 1:
            modes = modes.reshape(1, -1)
        
        M, N = modes.shape
        
        # Time axis
        t = np.arange(N) / fs
        
        # Frequency axis
        freq_bins = np.linspace(0, fs / 2, num_freq_bins)
        
        # Spectrum matrix
        spectrum = np.zeros((num_freq_bins, N))
        
        for mode in modes:
            amplitude, phase, inst_freq = self.hilbert.decompose_am_fm(mode, fs)
            
            # Map instantaneous frequency to bins
            for n in range(N):
                if inst_freq[n] >= 0 and inst_freq[n] < fs / 2:
                    bin_idx = int(inst_freq[n] / (fs / 2) * (num_freq_bins - 1))
                    bin_idx = min(bin_idx, num_freq_bins - 1)
                    spectrum[bin_idx, n] += amplitude[n] ** 2
        
        return t, freq_bins, spectrum
    
    def compute_marginal_spectrum(
        self,
        modes: np.ndarray,
        fs: float = 1.0,
        num_freq_bins: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute marginal Hilbert spectrum.
        
        Parameters:
        -----------
        modes : np.ndarray
            Array of modes
        fs : float
            Sampling frequency
        num_freq_bins : int
            Number of frequency bins
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Frequency axis and marginal spectrum
        """
        t, freq_bins, spectrum = self.compute_hilbert_spectrum(modes, fs, num_freq_bins)
        
        # Marginal spectrum: integrate over time
        marginal = np.sum(spectrum, axis=1)
        
        return freq_bins, marginal


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create test signal: AM-FM signal
    np.random.seed(42)
    N = 1000
    fs = 1000  # Hz
    t = np.arange(N) / fs
    
    # Carrier frequency with frequency modulation
    fc = 50  # Hz
    fm = 2   # Modulation frequency
    freq_deviation = 10  # Hz
    
    # Amplitude modulation
    am_freq = 1  # Hz
    amplitude = 1 + 0.5 * np.sin(2 * np.pi * am_freq * t)
    
    # FM signal
    phase = 2 * np.pi * fc * t + (freq_deviation / fm) * np.sin(2 * np.pi * fm * t)
    signal = amplitude * np.cos(phase)
    
    # Add some noise
    signal += 0.1 * np.random.randn(N)
    
    # Apply Hilbert transform
    hilbert = DiscreteHilbertTransform()
    
    # Get components
    recovered_amplitude = hilbert.get_envelope(signal)
    recovered_phase = hilbert.get_phase(signal)
    inst_freq = hilbert.get_instantaneous_frequency(signal, fs)
    
    # Check if it's an IMF
    is_imf, details = hilbert.is_imf(signal)
    print(f"Is IMF: {is_imf}")
    print(f"Details: {details}")
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    axes[0].plot(t, signal, 'b-', label='Signal')
    axes[0].plot(t, recovered_amplitude, 'r--', label='Envelope')
    axes[0].plot(t, -recovered_amplitude, 'r--')
    axes[0].set_title('Signal and Envelope')
    axes[0].legend()
    axes[0].set_xlabel('Time (s)')
    
    axes[1].plot(t, amplitude, 'b-', label='True Amplitude')
    axes[1].plot(t, recovered_amplitude, 'r--', label='Recovered Amplitude')
    axes[1].set_title('Amplitude Modulation')
    axes[1].legend()
    axes[1].set_xlabel('Time (s)')
    
    axes[2].plot(t, np.unwrap(phase), 'b-', label='True Phase')
    axes[2].plot(t, recovered_phase, 'r--', label='Recovered Phase')
    axes[2].set_title('Phase')
    axes[2].legend()
    axes[2].set_xlabel('Time (s)')
    
    # True instantaneous frequency
    true_inst_freq = fc + freq_deviation * np.cos(2 * np.pi * fm * t)
    axes[3].plot(t, true_inst_freq, 'b-', label='True Inst. Freq')
    axes[3].plot(t, inst_freq, 'r--', alpha=0.7, label='Recovered Inst. Freq')
    axes[3].set_title('Instantaneous Frequency')
    axes[3].legend()
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('hilbert_transform_example.png', dpi=150)
    print("\nSaved example to 'hilbert_transform_example.png'")
