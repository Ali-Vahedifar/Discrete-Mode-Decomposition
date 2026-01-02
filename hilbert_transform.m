%% Discrete Hilbert Transform
%  ===========================
%  
%  Implementation of the N-point Discrete Hilbert Transform as described in
%  Equations 6-8 of the IEEE INFOCOM 2025 paper.
%
%  Author: Ali Vahedi (Mohammad Ali Vahedifar)
%  Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
%  Email: av@ece.au.dk
%
%  IEEE INFOCOM 2025: "Discrete Mode Decomposition Meets Shapley Value:
%  Robust Signal Prediction in Tactile Internet"
%
%  The Hilbert transform is characterized by its DFT coefficients (Eq. 6):
%    H[k] = -j  for 1 ≤ k < N/2
%    H[k] = 0   for k = 0, N/2
%    H[k] = j   for N/2 < k ≤ N-1
%
%  And impulse response (Eq. 7):
%    h[n] = (1 - (-1)^n) / (π*n)  for n ≠ 0
%    h[0] = 0
%
%  The analytic signal is constructed as (Eq. 8):
%    z[n] = x[n] + j*H{x[n]}
%
%  Usage:
%    y = hilbert_transform(x)           % Hilbert transform
%    z = analytic_signal(x)             % Analytic signal
%    [amp, phase, freq] = inst_params(x, fs)  % Instantaneous parameters
%
%  Inputs:
%    x  - Input real discrete signal (N x 1 vector)
%    fs - Sampling frequency (for instantaneous frequency in Hz)
%
%  Outputs:
%    y     - Hilbert transform of x
%    z     - Analytic signal
%    amp   - Instantaneous amplitude
%    phase - Instantaneous phase
%    freq  - Instantaneous frequency

function y = hilbert_transform(x)
    % Discrete Hilbert Transform using DFT
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    x = x(:);  % Ensure column vector
    N = length(x);
    
    % Compute DFT
    X = fft(x);
    
    % Create Hilbert filter in frequency domain (Eq. 6)
    H = zeros(N, 1);
    
    if mod(N, 2) == 0  % Even length
        H(2:N/2) = -1j;           % 1 ≤ k < N/2
        H(N/2+1) = 0;             % k = N/2
        H(N/2+2:N) = 1j;          % N/2 < k ≤ N-1
    else  % Odd length
        H(2:ceil(N/2)) = -1j;
        H(ceil(N/2)+1:N) = 1j;
    end
    % H[0] = 0 (already initialized)
    
    % Apply filter and inverse DFT
    Y = X .* H;
    y = real(ifft(Y));
end

function z = analytic_signal(x)
    % Compute analytic signal (Eq. 8)
    % z[n] = x[n] + j*H{x[n]}
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    x = x(:);
    y = hilbert_transform(x);
    z = x + 1j * y;
end

function [amplitude, phase, frequency] = inst_params(x, fs)
    % Compute instantaneous parameters from analytic signal
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    %
    % Inputs:
    %   x  - Input signal
    %   fs - Sampling frequency (default: 1)
    %
    % Outputs:
    %   amplitude - Instantaneous amplitude A[n]
    %   phase     - Instantaneous phase φ[n]
    %   frequency - Instantaneous frequency ω[n] (in Hz if fs provided)
    
    if nargin < 2
        fs = 1;
    end
    
    x = x(:);
    N = length(x);
    
    % Compute analytic signal
    z = analytic_signal(x);
    
    % Instantaneous amplitude: A[n] = |z[n]|
    amplitude = abs(z);
    
    % Instantaneous phase: φ[n] = arg(z[n])
    phase = unwrap(angle(z));
    
    % Instantaneous frequency: ω[n] = φ[n+1] - φ[n]
    % Convert to Hz by multiplying by fs/(2*pi)
    phase_diff = diff(phase);
    frequency = [phase_diff(1); phase_diff] * fs / (2 * pi);
end

function h = hilbert_impulse_response(N)
    % Compute Hilbert transform impulse response (Eq. 7)
    % h[n] = (1 - (-1)^n) / (π*n) for n ≠ 0
    % h[0] = 0
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    n = (-(N-1)/2 : (N-1)/2)';
    h = zeros(size(n));
    
    % Non-zero indices
    nonzero = n ~= 0;
    h(nonzero) = (1 - (-1).^n(nonzero)) ./ (pi * n(nonzero));
    % h[0] = 0 (already initialized)
end
