%% DMD+SMV Demo Script
%  ====================
%  
%  Demonstration of Discrete Mode Decomposition and Shapley Mode Value
%  for haptic signal prediction in Tactile Internet.
%
%  Author: Ali Vahedi (Mohammad Ali Vahedifar)
%  Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
%  Email: av@ece.au.dk
%
%  IEEE INFOCOM 2025: "Discrete Mode Decomposition Meets Shapley Value:
%  Robust Signal Prediction in Tactile Internet"
%
%  This research was supported by:
%  - TOAST project (EU Horizon Europe, Grant No. 101073465)
%  - Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
%  - NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

clear all; close all; clc;

fprintf('================================================================\n');
fprintf('DMD+SMV Demo: Haptic Signal Prediction for Tactile Internet\n');
fprintf('================================================================\n');
fprintf('Author: Ali Vahedi (Mohammad Ali Vahedifar)\n');
fprintf('IEEE INFOCOM 2025\n');
fprintf('================================================================\n\n');

%% 1. Generate Synthetic Haptic Signal
fprintf('1. Generating synthetic haptic signal...\n');

% Parameters
fs = 1000;          % Sampling rate: 1 kHz (paper specification)
duration = 1;       % Duration in seconds
N = fs * duration;  % Number of samples
t = (0:N-1)' / fs;  % Time vector

% Create signal with multiple frequency components
% Simulating haptic interaction: position, velocity, force
f1 = 5;   % Low frequency component (slow movement)
f2 = 15;  % Medium frequency (fine manipulation)
f3 = 35;  % High frequency (vibration feedback)

signal_clean = sin(2*pi*f1*t) + 0.6*sin(2*pi*f2*t) + 0.3*sin(2*pi*f3*t);

% Add noise
alpha = 0.05;  % Noise variance
noise = sqrt(alpha) * randn(N, 1);
signal = signal_clean + noise;

fprintf('   Signal length: %d samples\n', N);
fprintf('   Sampling rate: %d Hz\n', fs);
fprintf('   Frequency components: %d, %d, %d Hz\n', f1, f2, f3);
fprintf('   Noise variance (alpha): %.4f\n\n', alpha);

%% 2. Apply Discrete Mode Decomposition
fprintf('2. Applying Discrete Mode Decomposition (DMD)...\n');

% DMD parameters
dmd_params = struct();
dmd_params.epsilon1 = 1e-6;
dmd_params.epsilon2 = 1e-6;
dmd_params.tau1 = 0.1;
dmd_params.tau2 = 0.1;
dmd_params.kappa1 = 1e-3;
dmd_params.kappa2 = 1e-3;
dmd_params.max_modes = 5;
dmd_params.max_inner_iter = 200;
dmd_params.verbose = true;

% Run DMD
[modes, center_freqs, residual, dmd_info] = DMD(signal, alpha, dmd_params);

M = size(modes, 1);
fprintf('\n   Extracted %d modes\n', M);
fprintf('   Center frequencies: ');
for i = 1:M
    freq_hz = center_freqs(i) * fs / (2*pi);
    fprintf('%.1f Hz  ', freq_hz);
end
fprintf('\n\n');

%% 3. Apply Shapley Mode Value
fprintf('3. Computing Shapley Mode Values (SMV)...\n');

% Simple predictor function (sum of selected modes)
predictor = @(selected_modes, X) sum(selected_modes, 1)';

% Validation data
X_val = t;
y_val = signal_clean;

% SMV parameters
smv_params = struct();
smv_params.tolerance = 0.01;
smv_params.max_iterations = 500;
smv_params.epsilon3 = 0.01;
smv_params.metric = 'accuracy';
smv_params.verbose = true;

% Run SMV
[shapley_values, rankings, smv_info] = SMV(modes, center_freqs, predictor, X_val, y_val, smv_params);

fprintf('\n   Mode Rankings (by importance):\n');
for i = 1:M
    mode_idx = rankings(i);
    freq_hz = center_freqs(mode_idx) * fs / (2*pi);
    fprintf('   Rank %d: Mode %d (%.1f Hz), Shapley value = %.4f\n', ...
            i, mode_idx, freq_hz, shapley_values(mode_idx));
end
fprintf('\n');

%% 4. Select Top-K Modes
fprintf('4. Selecting top-K modes for prediction...\n');

K = smv_info.K;
top_modes = rankings(1:K);

fprintf('   Selected %d modes: %s\n', K, mat2str(top_modes'));

% Reconstruct with selected modes
signal_reconstructed_full = sum(modes, 1)' + residual;
signal_reconstructed_topK = sum(modes(top_modes, :), 1)';

% Compute errors
mse_full = mean((signal_clean - signal_reconstructed_full).^2);
mse_topK = mean((signal_clean - signal_reconstructed_topK).^2);

fprintf('   MSE (all modes): %.6f\n', mse_full);
fprintf('   MSE (top-K modes): %.6f\n', mse_topK);
fprintf('   Compression ratio: %.1fx (using %d/%d modes)\n\n', M/K, K, M);

%% 5. Evaluate Prediction Accuracy
fprintf('5. Evaluating prediction performance...\n');

% Paper's accuracy metric: Accuracy = (1 - MAE/range) * 100
mae_full = mean(abs(signal_clean - signal_reconstructed_full));
mae_topK = mean(abs(signal_clean - signal_reconstructed_topK));
data_range = max(signal_clean) - min(signal_clean);

accuracy_full = (1 - mae_full / data_range) * 100;
accuracy_topK = (1 - mae_topK / data_range) * 100;

% PSNR
psnr_full = 10 * log10(max(signal_clean)^2 / mse_full);
psnr_topK = 10 * log10(max(signal_clean)^2 / mse_topK);

fprintf('   Accuracy (all modes): %.2f%%\n', accuracy_full);
fprintf('   Accuracy (top-K modes): %.2f%%\n', accuracy_topK);
fprintf('   PSNR (all modes): %.2f dB\n', psnr_full);
fprintf('   PSNR (top-K modes): %.2f dB\n\n', psnr_topK);

%% 6. Visualization
fprintf('6. Generating visualizations...\n');

figure('Position', [50, 50, 1400, 900], 'Color', 'w');

% Original and noisy signal
subplot(3, 2, 1);
plot(t*1000, signal_clean, 'b-', 'LineWidth', 1.5);
hold on;
plot(t*1000, signal, 'r-', 'LineWidth', 0.5, 'Color', [1, 0.6, 0.6]);
xlabel('Time (ms)');
ylabel('Amplitude');
title('Original Signal');
legend('Clean', 'Noisy', 'Location', 'best');
grid on;
xlim([0, 200]);

% Extracted modes
subplot(3, 2, 2);
colors = lines(M);
hold on;
for i = 1:M
    freq_hz = center_freqs(i) * fs / (2*pi);
    plot(t*1000, modes(i,:) + (M-i)*2, 'Color', colors(i,:), 'LineWidth', 1.2);
end
xlabel('Time (ms)');
ylabel('Modes (offset for visibility)');
title(sprintf('Extracted Modes (M=%d)', M));
xlim([0, 200]);
grid on;

% Shapley values
subplot(3, 2, 3);
bar(shapley_values, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('Mode Index');
ylabel('Shapley Value');
title('Shapley Mode Values');
grid on;
for i = 1:M
    freq_hz = center_freqs(i) * fs / (2*pi);
    text(i, shapley_values(i) + 0.01, sprintf('%.0fHz', freq_hz), ...
         'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Reconstruction comparison
subplot(3, 2, 4);
plot(t*1000, signal_clean, 'b-', 'LineWidth', 1.5);
hold on;
plot(t*1000, signal_reconstructed_topK, 'g--', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Amplitude');
title(sprintf('Reconstruction with Top-%d Modes (Acc: %.1f%%)', K, accuracy_topK));
legend('Original', 'Reconstructed', 'Location', 'best');
grid on;
xlim([0, 200]);

% Frequency spectrum
subplot(3, 2, 5);
f_axis = (0:N-1) * fs / N;
signal_fft = abs(fft(signal));
signal_fft = signal_fft(1:N/2);
f_axis = f_axis(1:N/2);
plot(f_axis, signal_fft, 'b-', 'LineWidth', 1);
hold on;
for i = 1:M
    freq_hz = center_freqs(i) * fs / (2*pi);
    xline(freq_hz, '--', sprintf('Mode %d', i), 'Color', colors(i,:), 'LineWidth', 1.5);
end
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum with Mode Centers');
xlim([0, 100]);
grid on;

% Convergence history (SMV)
subplot(3, 2, 6);
if isfield(smv_info, 'shapley_history')
    plot(smv_info.shapley_history', 'LineWidth', 1.2);
    xlabel('Iteration');
    ylabel('Shapley Value');
    title('SMV Convergence');
    legend(arrayfun(@(i) sprintf('Mode %d', i), 1:M, 'UniformOutput', false), ...
           'Location', 'best');
    grid on;
end

sgtitle({'DMD+SMV for Haptic Signal Prediction', ...
         'Author: Ali Vahedi (Mohammad Ali Vahedifar) - IEEE INFOCOM 2025'}, ...
         'FontWeight', 'bold');

% Save figure
saveas(gcf, 'dmd_smv_demo_results.png');
fprintf('   Saved figure to: dmd_smv_demo_results.png\n');

%% 7. Summary
fprintf('\n================================================================\n');
fprintf('SUMMARY\n');
fprintf('================================================================\n');
fprintf('DMD extracted %d modes from the signal\n', M);
fprintf('SMV identified top %d most important modes\n', K);
fprintf('Prediction accuracy: %.2f%% (paper reports 98.9%% at W=1)\n', accuracy_topK);
fprintf('PSNR: %.2f dB (paper reports ~29.5 dB)\n', psnr_topK);
fprintf('================================================================\n');
fprintf('For more information, see the IEEE INFOCOM 2025 paper:\n');
fprintf('"Discrete Mode Decomposition Meets Shapley Value:\n');
fprintf(' Robust Signal Prediction in Tactile Internet"\n');
fprintf('================================================================\n');
