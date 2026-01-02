%% Discrete Mode Decomposition (DMD) - Main Implementation
%  =========================================================
%  
%  Implementation of the DMD algorithm as described in:
%  "Discrete Mode Decomposition Meets Shapley Value: Robust Signal 
%  Prediction in Tactile Internet" - IEEE INFOCOM 2025
%
%  Author: Ali Vahedi (Mohammad Ali Vahedifar)
%  Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
%  Email: av@ece.au.dk
%
%  This research was supported by:
%  - TOAST project (EU Horizon Europe, Grant No. 101073465)
%  - Danish Council for Independent Research eTouch (Grant No. 1127-00339B)
%  - NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)
%
%  Usage:
%    [modes, center_freqs, residual] = DMD(signal, alpha)
%    [modes, center_freqs, residual] = DMD(signal, alpha, params)
%
%  Inputs:
%    signal - Input discrete-time signal (N x 1 vector)
%    alpha  - Noise variance parameter
%    params - (Optional) Structure with algorithm parameters
%
%  Outputs:
%    modes        - Extracted modes (M x N matrix)
%    center_freqs - Center frequencies of each mode (M x 1 vector)
%    residual     - Unprocessed signal component (N x 1 vector)

function [modes, center_freqs, residual, info] = DMD(signal, alpha, params)
    
    %% Default parameters
    if nargin < 3
        params = struct();
    end
    
    % Algorithm parameters (from paper)
    params = set_default(params, 'epsilon1', 1e-6);      % Regularization for mode overlap
    params = set_default(params, 'epsilon2', 1e-6);      % Regularization for unprocessed signal
    params = set_default(params, 'tau1', 0.1);           % Step size for rho update (Eq. 31)
    params = set_default(params, 'tau2', 0.1);           % Step size for mu update (Eq. 32)
    params = set_default(params, 'kappa1', 1e-3);        % Inner loop convergence threshold
    params = set_default(params, 'kappa2', 1e-3);        % Outer loop convergence threshold
    params = set_default(params, 'max_modes', 10);       % Maximum number of modes
    params = set_default(params, 'max_inner_iter', 500); % Max inner ADMM iterations
    params = set_default(params, 'rho_init', 1.0);       % Initial penalty parameter
    params = set_default(params, 'mu_init', 0.0);        % Initial inequality multiplier
    params = set_default(params, 'verbose', true);       % Print progress
    
    %% Initialize
    signal = signal(:);  % Ensure column vector
    N = length(signal);
    
    % Frequency axis
    omega = (0:N-1)' * 2 * pi / N;
    omega_positive = omega(1:floor(N/2)+1);
    
    % Storage
    modes = [];
    center_freqs = [];
    convergence_history = [];
    
    % Current residual (unprocessed signal)
    residual = signal;
    
    % Signal spectrum
    X = fft(signal);
    
    if params.verbose
        fprintf('========================================\n');
        fprintf('Discrete Mode Decomposition (DMD)\n');
        fprintf('Author: Ali Vahedi (Mohammad Ali Vahedifar)\n');
        fprintf('IEEE INFOCOM 2025\n');
        fprintf('========================================\n');
        fprintf('Signal length: %d samples\n', N);
        fprintf('Noise variance (alpha): %.4f\n', alpha);
        fprintf('----------------------------------------\n');
    end
    
    %% Main loop - Extract modes iteratively (Algorithm 1)
    M = 0;  % Mode counter
    
    while M < params.max_modes
        M = M + 1;
        
        if params.verbose
            fprintf('\nExtracting mode %d...\n', M);
        end
        
        % Initialize current mode
        [U_M, omega_M] = initialize_mode(residual, N, omega);
        
        % ADMM variables
        rho = params.rho_init * ones(N, 1);
        mu = params.mu_init;
        theta = zeros(N, 1);
        
        % Inner ADMM loop
        for n_iter = 1:params.max_inner_iter
            U_M_prev = U_M;
            
            % Update U_M (Eq. 24)
            U_M = update_mode(X, modes, center_freqs, residual, ...
                              omega_M, rho, theta, alpha, params, N, omega);
            
            % Update omega_M (Eq. 27)
            omega_M = update_center_frequency(U_M, omega_positive);
            
            % Update X_u (unprocessed signal in frequency domain)
            X_u = update_unprocessed(X, modes, U_M, omega_M, rho, mu, ...
                                     theta, alpha, params, N, omega);
            
            % Update rho (Eq. 31)
            sum_modes = zeros(N, 1);
            for i = 1:size(modes, 1)
                sum_modes = sum_modes + fft(modes(i, :)');
            end
            sum_modes = sum_modes + U_M;
            rho = rho + params.tau1 * abs(X - sum_modes);
            
            % Update mu (Eq. 32) - Energy-based constraint with max(0, ...)
            X_u_energy = sum(abs(X_u).^2);
            if ~isempty(modes)
                mode_energies = sum(abs(fft(modes')).^2, 1);
                U_min_energy = min(mode_energies);
            else
                U_min_energy = sum(abs(U_M).^2);
            end
            mu = max(0, mu + params.tau2 * (X_u_energy - U_min_energy));
            
            % Check convergence (normalized, as in Algorithm 1)
            conv_metric = norm(U_M - U_M_prev)^2 / (norm(U_M_prev)^2 + 1e-10);
            
            if conv_metric < params.kappa1
                if params.verbose
                    fprintf('  Inner loop converged at iteration %d (metric: %.2e)\n', ...
                            n_iter, conv_metric);
                end
                break;
            end
        end
        
        % Convert mode to time domain
        u_M = real(ifft(U_M));
        
        % Store mode
        modes = [modes; u_M'];
        center_freqs = [center_freqs; omega_M];
        
        % Update residual
        residual = signal - sum(modes, 1)';
        
        % Check outer loop convergence
        reconstruction_error = norm(residual)^2 / N;
        outer_metric = abs(alpha - reconstruction_error) / alpha;
        
        convergence_history = [convergence_history; struct(...
            'mode', M, ...
            'inner_iterations', n_iter, ...
            'reconstruction_error', reconstruction_error, ...
            'outer_metric', outer_metric)];
        
        if params.verbose
            fprintf('  Mode %d: center_freq = %.4f rad/sample\n', M, omega_M);
            fprintf('  Reconstruction error: %.6f\n', reconstruction_error);
            fprintf('  Outer metric: %.4f (threshold: %.4f)\n', outer_metric, params.kappa2);
        end
        
        if outer_metric < params.kappa2
            if params.verbose
                fprintf('\nOuter loop converged after %d modes.\n', M);
            end
            break;
        end
    end
    
    %% Output info
    info = struct();
    info.num_modes = M;
    info.convergence_history = convergence_history;
    info.final_reconstruction_error = norm(signal - sum(modes, 1)')^2 / N;
    info.params = params;
    
    if params.verbose
        fprintf('\n========================================\n');
        fprintf('DMD Complete\n');
        fprintf('Extracted %d modes\n', M);
        fprintf('Final reconstruction error: %.6f\n', info.final_reconstruction_error);
        fprintf('========================================\n');
    end
end

%% Helper Functions

function params = set_default(params, field, value)
    % Set default parameter value if not specified
    if ~isfield(params, field)
        params.(field) = value;
    end
end

function [U_M, omega_M] = initialize_mode(residual, N, omega)
    % Initialize mode using FFT of residual
    R = fft(residual);
    
    % Find dominant frequency
    R_positive = abs(R(1:floor(N/2)+1));
    [~, idx] = max(R_positive);
    omega_M = omega(idx);
    
    % Initialize mode spectrum
    U_M = R .* exp(-(omega - omega_M).^2 / 0.1);
end

function U_M = update_mode(X, modes, center_freqs, residual, omega_M, ...
                           rho, theta, alpha, params, N, omega)
    % Update U_M using Equation 24 from the paper
    %
    % U_M^{n+1}(ω) = [ρ(ω)/2 * Q(ω)] / [ρ(ω)/2 + Σ|β_i(ω)|² + (2/π)sin²(ω-ω_M)]
    
    % Compute Q(ω) = X(ω) - Σ U_i(ω) - X_u(ω) + Θ(ω)
    sum_prev_modes = zeros(N, 1);
    for i = 1:size(modes, 1)
        sum_prev_modes = sum_prev_modes + fft(modes(i, :)');
    end
    
    X_u = fft(residual);
    Theta = theta;
    
    Q = X - sum_prev_modes - X_u + Theta;
    
    % Compute sum of β_i filters (Eq. 12)
    beta_sum = zeros(N, 1);
    for i = 1:length(center_freqs)
        omega_i = center_freqs(i);
        beta_i = 1 ./ (alpha * (omega - omega_i).^2 + params.epsilon1);
        beta_sum = beta_sum + abs(beta_i).^2;
    end
    
    % Spectral compactness term
    sin_sq = sin(omega - omega_M).^2;
    
    % Update equation (Eq. 24)
    numerator = (rho / 2) .* Q;
    denominator = (rho / 2) + beta_sum + (2 / pi) * sin_sq;
    
    U_M = numerator ./ (denominator + 1e-10);
end

function omega_M = update_center_frequency(U_M, omega_positive)
    % Update center frequency using Equation 27 from the paper
    %
    % ω_M^{n+1} = ∫ω|U_M(ω)|²dω / ∫|U_M(ω)|²dω
    
    N = length(U_M);
    U_M_positive = U_M(1:floor(N/2)+1);
    
    U_M_abs_sq = abs(U_M_positive).^2;
    
    numerator = sum(omega_positive .* U_M_abs_sq);
    denominator = sum(U_M_abs_sq) + 1e-10;
    
    omega_M = numerator / denominator;
end

function X_u = update_unprocessed(X, modes, U_M, omega_M, rho, mu, ...
                                   theta, alpha, params, N, omega)
    % Update unprocessed signal using Equation 30 from the paper
    %
    % X_u^{n+1}(ω) = [ρ(ω) * Q̃(ω)] / [2|β_M(ω)|² + 2μ + ρ(ω)]
    
    % Compute Q̃(ω) = X(ω) - Σ U_i(ω) + Θ(ω)
    sum_modes = zeros(N, 1);
    for i = 1:size(modes, 1)
        sum_modes = sum_modes + fft(modes(i, :)');
    end
    sum_modes = sum_modes + U_M;
    
    Q_tilde = X - sum_modes + theta;
    
    % β_M filter (Eq. 14)
    beta_M = 1 ./ (alpha * (omega - omega_M).^2 + params.epsilon2);
    
    % Update equation (Eq. 30) - includes 2*mu term
    numerator = rho .* Q_tilde;
    denominator = 2 * abs(beta_M).^2 + 2 * mu + rho;
    
    X_u = numerator ./ (denominator + 1e-10);
end
