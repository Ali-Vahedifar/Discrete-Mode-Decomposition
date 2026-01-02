%% Discrete Wiener Filtering
%  ==========================
%  
%  Implementation of Discrete Wiener Filtering as described in
%  Section IV-B (Equations 2-5) of the IEEE INFOCOM 2025 paper.
%
%  Author: Ali Vahedi (Mohammad Ali Vahedifar)
%  Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
%  Email: av@ece.au.dk
%
%  IEEE INFOCOM 2025: "Discrete Mode Decomposition Meets Shapley Value:
%  Robust Signal Prediction in Tactile Internet"
%
%  Mathematical Background:
%  ------------------------
%  Given observed signal y[n] = x[n] + η[n], where η[n] ~ N(0, α)
%
%  The denoising problem is formulated as Tikhonov regularization (Eq. 3):
%    min_x { ||x[n] - y[n]||²₂ + α ||∇x[n]||²₂ }
%
%  Solution in frequency domain (Eq. 4):
%    X(ω) = Y(ω) / (1 + α|ω|²)
%
%  This corresponds to convolution with a Wiener filter with
%  low-pass power spectrum prior 1/|ω|².
%
%  Usage:
%    x_denoised = wiener_filter(y, alpha)
%    [x_denoised, H] = wiener_filter(y, alpha)
%
%  Inputs:
%    y     - Noisy observed signal (N x 1 vector)
%    alpha - Noise variance parameter
%
%  Outputs:
%    x_denoised - Denoised signal estimate
%    H          - Wiener filter frequency response

function [x_denoised, H] = wiener_filter(y, alpha)
    % Discrete Wiener Filter for signal denoising
    %
    % Implements Equation 4 from the paper:
    % X(ω) = Y(ω) / (1 + α|ω|²)
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    y = y(:);  % Ensure column vector
    N = length(y);
    
    % Frequency axis: ω ∈ [-π, π]
    if mod(N, 2) == 0
        omega = (-N/2 : N/2-1)' * 2 * pi / N;
    else
        omega = (-(N-1)/2 : (N-1)/2)' * 2 * pi / N;
    end
    omega = fftshift(omega);  % Align with FFT output
    
    % Compute FFT of noisy signal
    Y = fft(y);
    
    % Wiener filter frequency response (Eq. 4)
    % H(ω) = 1 / (1 + α|ω|²)
    H = 1 ./ (1 + alpha * abs(omega).^2);
    
    % Apply filter in frequency domain
    X = Y .* H;
    
    % Inverse FFT to get denoised signal
    x_denoised = real(ifft(X));
end

function [x_denoised, H] = wiener_filter_adaptive(y, alpha, method)
    % Adaptive Wiener Filter with different estimation methods
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    %
    % Inputs:
    %   y      - Noisy signal
    %   alpha  - Noise variance (or 'estimate' to auto-estimate)
    %   method - 'standard', 'local', 'frequency'
    %
    % Outputs:
    %   x_denoised - Denoised signal
    %   H          - Filter response
    
    if nargin < 3
        method = 'standard';
    end
    
    y = y(:);
    N = length(y);
    
    % Estimate noise variance if not provided
    if ischar(alpha) && strcmp(alpha, 'estimate')
        % Estimate from high-frequency components
        Y = fft(y);
        high_freq_idx = round(0.8*N):N;
        alpha = mean(abs(Y(high_freq_idx)).^2);
    end
    
    switch method
        case 'standard'
            % Standard Wiener filter (Eq. 4)
            [x_denoised, H] = wiener_filter(y, alpha);
            
        case 'local'
            % Local adaptive Wiener filter
            window_size = min(32, N);
            x_denoised = zeros(N, 1);
            H = zeros(N, 1);
            
            for i = 1:N
                % Extract local window
                idx_start = max(1, i - floor(window_size/2));
                idx_end = min(N, i + floor(window_size/2));
                local_y = y(idx_start:idx_end);
                
                % Local statistics
                local_mean = mean(local_y);
                local_var = var(local_y);
                
                % Wiener estimate
                if local_var > alpha
                    x_denoised(i) = local_mean + (local_var - alpha) / local_var * (y(i) - local_mean);
                else
                    x_denoised(i) = local_mean;
                end
            end
            
        case 'frequency'
            % Frequency-dependent Wiener filter
            Y = fft(y);
            
            % Estimate signal power spectrum
            P_y = abs(Y).^2;
            P_noise = alpha * N;  % Noise power
            P_signal = max(P_y - P_noise, 0);
            
            % Wiener filter
            H = P_signal ./ (P_signal + P_noise + 1e-10);
            X = Y .* H;
            x_denoised = real(ifft(X));
            
        otherwise
            error('Unknown method: %s', method);
    end
end

function alpha = estimate_noise_variance(y, method)
    % Estimate noise variance from signal
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    %
    % Methods:
    %   'mad'     - Median Absolute Deviation (robust)
    %   'highfreq'- From high-frequency components
    %   'diff'    - From signal differences
    
    if nargin < 2
        method = 'mad';
    end
    
    y = y(:);
    N = length(y);
    
    switch method
        case 'mad'
            % Robust estimation using MAD
            % Based on wavelet-domain noise estimation
            detail = diff(y);
            mad_val = median(abs(detail - median(detail)));
            alpha = (mad_val / 0.6745)^2;  % Scale for Gaussian
            
        case 'highfreq'
            % Estimate from high-frequency FFT components
            Y = fft(y);
            high_freq_idx = round(0.75*N):N;
            alpha = mean(abs(Y(high_freq_idx)).^2) / N;
            
        case 'diff'
            % Estimate from second differences
            d2 = diff(diff(y));
            alpha = var(d2) / 6;  % Adjust for differencing
            
        otherwise
            error('Unknown method: %s', method);
    end
end

function demo_wiener_filter()
    % Demonstration of Wiener filter
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    fprintf('Wiener Filter Demonstration\n');
    fprintf('Author: Ali Vahedi (Mohammad Ali Vahedifar)\n');
    fprintf('IEEE INFOCOM 2025\n\n');
    
    % Create test signal
    N = 1000;
    t = (0:N-1)' / N;
    
    % Clean signal: sum of sinusoids
    x_clean = sin(2*pi*5*t) + 0.5*sin(2*pi*15*t) + 0.3*sin(2*pi*30*t);
    
    % Add noise
    alpha = 0.1;  % Noise variance
    noise = sqrt(alpha) * randn(N, 1);
    y_noisy = x_clean + noise;
    
    % Apply Wiener filter
    [x_denoised, H] = wiener_filter(y_noisy, alpha);
    
    % Compute errors
    noisy_error = norm(y_noisy - x_clean)^2 / N;
    denoised_error = norm(x_denoised - x_clean)^2 / N;
    
    fprintf('Noise variance (alpha): %.4f\n', alpha);
    fprintf('MSE (noisy):    %.6f\n', noisy_error);
    fprintf('MSE (denoised): %.6f\n', denoised_error);
    fprintf('Improvement:    %.2f dB\n', 10*log10(noisy_error/denoised_error));
    
    % Plot results
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(3, 1, 1);
    plot(t, x_clean, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t, y_noisy, 'r-', 'LineWidth', 0.5, 'Color', [1, 0.5, 0.5]);
    legend('Clean', 'Noisy');
    title('Original and Noisy Signal');
    xlabel('Time');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 1, 2);
    plot(t, x_clean, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t, x_denoised, 'g-', 'LineWidth', 1.5);
    legend('Clean', 'Denoised');
    title('Clean and Denoised Signal');
    xlabel('Time');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 1, 3);
    omega = (-N/2:N/2-1)' * 2 * pi / N;
    plot(fftshift(omega), fftshift(H), 'b-', 'LineWidth', 1.5);
    title('Wiener Filter Frequency Response');
    xlabel('Frequency (rad/sample)');
    ylabel('|H(ω)|');
    grid on;
    xlim([-pi, pi]);
end
