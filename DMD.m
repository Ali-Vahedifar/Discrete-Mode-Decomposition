%% Discrete Mode Decomposition (DMD) - Exact Implementation
%  ==========================================================
%  
%  EXACT implementation of DMD algorithm as described in:
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
%  Key Equations from Paper:
%  -------------------------
%  Eq. 10 (T1): Spectral compactness - ||∂_n[A_n * u_M] e^{-jω_M n}||²
%  Eq. 11-12 (T2): Minimum overlap with previous modes - Σ||β_i * u_M||²
%  Eq. 13-14 (T3): Minimum overlap with unprocessed - ||β_M * x_u||²
%  Eq. 15: Reconstruction constraint - x[n] = Σu_k[n]
%  Eq. 16: Energy bound - ||x_u||² ≤ ||u_min||²
%  
%  Eq. 18: Augmented Lagrangian
%  Eq. 19: Frequency domain Lagrangian
%  Eq. 24: U_M update
%  Eq. 27: ω_M update  
%  Eq. 30: X_u update (with 2μ term)
%  Eq. 31: ρ update
%  Eq. 32: μ update (with max and integral)
%
%  Usage:
%    [modes, center_freqs, residual, info] = DMD(signal, alpha)
%    [modes, center_freqs, residual, info] = DMD(signal, alpha, params)

function [modes, center_freqs, residual, info] = DMD(signal, alpha, params)
    
    %% Default parameters
    if nargin < 3
        params = struct();
    end
    
    % Algorithm parameters (from paper Section IV)
    params = set_default(params, 'epsilon1', 1e-6);      % ε₁: Regularization for β_i (Eq. 12)
    params = set_default(params, 'epsilon2', 1e-6);      % ε₂: Regularization for β_M (Eq. 14)
    params = set_default(params, 'tau1', 0.1);           % τ₁: Step size for ρ update (Eq. 31)
    params = set_default(params, 'tau2', 0.1);           % τ₂: Step size for μ update (Eq. 32)
    params = set_default(params, 'kappa1', 1e-3);        % κ₁: Inner convergence (Algorithm 1)
    params = set_default(params, 'kappa2', 1e-3);        % κ₂: Outer convergence (Algorithm 1)
    params = set_default(params, 'max_modes', 10);       % Maximum number of modes
    params = set_default(params, 'max_inner_iter', 500); % Max inner ADMM iterations
    params = set_default(params, 'rho_init', 1.0);       % Initial ρ
    params = set_default(params, 'mu_init', 0.0);        % Initial μ
    params = set_default(params, 'verbose', true);       % Print progress
    
    %% Initialize
    signal = signal(:);  % Ensure column vector
    N = length(signal);
    
    % Frequency axis: ω ∈ [0, 2π)
    omega = (0:N-1)' * 2 * pi / N;
    
    % For integrals over [0, π] (positive frequencies)
    N_half = floor(N/2) + 1;
    omega_positive = omega(1:N_half);
    d_omega = 2 * pi / N;  % Frequency resolution
    
    % Storage
    modes = [];
    center_freqs = [];
    convergence_history = [];
    
    % Current unprocessed signal
    x_u = zeros(N, 1);
    
    % Signal spectrum
    X = fft(signal);
    
    if params.verbose
        fprintf('========================================\n');
        fprintf('Discrete Mode Decomposition (DMD)\n');
        fprintf('Author: Ali Vahedi (Mohammad Ali Vahedifar)\n');
        fprintf('IEEE INFOCOM 2025\n');
        fprintf('========================================\n');
        fprintf('Signal length: N = %d samples\n', N);
        fprintf('Noise variance (α): %.4f\n', alpha);
        fprintf('----------------------------------------\n');
    end
    
    %% Main loop - Extract modes iteratively (Algorithm 1)
    % REPEAT until outer convergence
    M = 0;  % Mode counter
    
    while M < params.max_modes
        M = M + 1;
        
        if params.verbose
            fprintf('\nExtracting mode M = %d...\n', M);
        end
        
        %% Initialize for new mode (Algorithm 1, line 3)
        % Initialize: u¹_M, ω¹_M, ρ¹, μ¹, n←0
        [U_M, omega_M] = initialize_mode(signal, modes, N, omega);
        
        rho = params.rho_init * ones(N, 1);  % ρ(ω)
        mu = params.mu_init;                  % μ (scalar)
        Theta = zeros(N, 1);                  % Θ(ω) scaled dual variable
        
        n_iter = 0;
        
        %% Inner ADMM loop (Algorithm 1, lines 6-11)
        % REPEAT until ||U_M^{n+1} - U_M^n||²/||U_M^n||² ≤ κ₁
        while n_iter < params.max_inner_iter
            n_iter = n_iter + 1;
            U_M_prev = U_M;
            
            %% Step 1: Update U_M(ω) with Eq. 24
            % U_M^{n+1}(ω) = [ρ(ω)/2 · Q(ω)] / [ρ(ω)/2 + Σ|β_i(ω)|² + (2/π)sin²(ω-ω_M)]
            % where Q(ω) = X(ω) - Σ_{i=1}^{M-1} U_i(ω) - X_u(ω) + Θ(ω)
            
            % Sum of previous modes in frequency domain
            sum_prev_U = zeros(N, 1);
            for i = 1:size(modes, 1)
                sum_prev_U = sum_prev_U + fft(modes(i, :)');
            end
            
            % X_u in frequency domain
            X_u = fft(x_u);
            
            % Q(ω) - Eq. 22
            Q = X - sum_prev_U - X_u + Theta;
            
            % Sum of |β_i(ω)|² for i = 1, ..., M-1 (Eq. 12)
            beta_sq_sum = zeros(N, 1);
            for i = 1:length(center_freqs)
                omega_i = center_freqs(i);
                % β_i(ω) = 1 / [α(ω - ω_i)² + ε₁]
                beta_i = 1 ./ (alpha * (omega - omega_i).^2 + params.epsilon1);
                beta_sq_sum = beta_sq_sum + abs(beta_i).^2;
            end
            
            % Spectral compactness term: (2/π) sin²(ω - ω_M)
            sin_sq_term = (2/pi) * sin(omega - omega_M).^2;
            
            % Update U_M (Eq. 24)
            numerator = (rho/2) .* Q;
            denominator = (rho/2) + beta_sq_sum + sin_sq_term;
            U_M = numerator ./ (denominator + 1e-10);
            
            %% Step 2: Update ω_M with Eq. 27
            % ω_M^{n+1} = ∫₀^π ω|U_M(ω)|² dω / ∫₀^π |U_M(ω)|² dω
            
            U_M_positive = U_M(1:N_half);
            U_M_abs_sq = abs(U_M_positive).^2;
            
            % Numerical integration over [0, π]
            numerator_omega = sum(omega_positive .* U_M_abs_sq) * d_omega;
            denominator_omega = sum(U_M_abs_sq) * d_omega + 1e-10;
            omega_M = numerator_omega / denominator_omega;
            
            %% Step 3: Update X_u(ω) with Eq. 30 (implicitly through x_u)
            % X_u^{n+1}(ω) = [ρ(ω) · Q̃(ω)] / [2|β_M(ω)|² + 2μ + ρ(ω)]
            % where Q̃(ω) = X(ω) - Σ_{i=1}^M U_i(ω) + Θ(ω)
            
            % Q̃(ω) - Eq. 28
            Q_tilde = X - sum_prev_U - U_M + Theta;
            
            % β_M(ω) = 1 / [α(ω - ω_M)² + ε₂] (Eq. 14)
            beta_M = 1 ./ (alpha * (omega - omega_M).^2 + params.epsilon2);
            
            % Update X_u (Eq. 30) - NOTE: includes 2μ in denominator
            numerator_Xu = rho .* Q_tilde;
            denominator_Xu = 2 * abs(beta_M).^2 + 2 * mu + rho;
            X_u = numerator_Xu ./ (denominator_Xu + 1e-10);
            
            % Convert to time domain
            x_u = real(ifft(X_u));
            
            %% Step 4: Update ρ(ω) with Eq. 31
            % ρ^{n+1}(ω) = ρ^n(ω) + τ₁ · (X(ω) - Σ_{i=1}^M U_i^{n+1}(ω))
            
            sum_all_U = sum_prev_U + U_M;
            rho = rho + params.tau1 * (X - sum_all_U);
            rho = abs(rho);  % Ensure positive
            
            %% Step 5: Update μ with Eq. 32
            % μ^{n+1} = max(0, μ^n + τ₂ · ∫₀^π (||X_u(ω)||² - ||U_min(ω)||²) dω)
            
            % Energy of X_u over [0, π]
            X_u_energy = sum(abs(X_u(1:N_half)).^2) * d_omega;
            
            % Find U_min = argmin_{u ∈ U} ||u||₂
            if ~isempty(modes)
                mode_energies = zeros(size(modes, 1), 1);
                for i = 1:size(modes, 1)
                    U_i = fft(modes(i, :)');
                    mode_energies(i) = sum(abs(U_i(1:N_half)).^2) * d_omega;
                end
                U_min_energy = min(mode_energies);
            else
                U_min_energy = sum(abs(U_M(1:N_half)).^2) * d_omega;
            end
            
            % Update μ (Eq. 32) - with max(0, ...)
            mu = max(0, mu + params.tau2 * (X_u_energy - U_min_energy));
            
            %% Step 6: Update Θ(ω) - scaled dual variable
            % Θ^{n+1}(ω) = Θ^n(ω) + (X(ω) - Σ U_i(ω) - X_u(ω))
            Theta = Theta + (X - sum_all_U - X_u);
            
            %% Check inner convergence (Algorithm 1, line 11)
            % ||U_M^{n+1} - U_M^n||² / ||U_M^n||² ≤ κ₁
            conv_metric = norm(U_M - U_M_prev)^2 / (norm(U_M_prev)^2 + 1e-10);
            
            if conv_metric < params.kappa1
                if params.verbose
                    fprintf('  Inner converged: iter %d, metric = %.2e < κ₁ = %.2e\n', ...
                            n_iter, conv_metric, params.kappa1);
                end
                break;
            end
        end
        
        %% Store extracted mode
        u_M = real(ifft(U_M));
        modes = [modes; u_M'];
        center_freqs = [center_freqs; omega_M];
        
        %% Update residual
        residual = signal - sum(modes, 1)' - x_u;
        
        %% Check outer convergence (Algorithm 1, line 12)
        % |α - (1/N)||x - Σ u_i||²| / α ≤ κ₂
        reconstruction_error = norm(signal - sum(modes, 1)')^2 / N;
        outer_metric = abs(alpha - reconstruction_error) / alpha;
        
        convergence_history = [convergence_history; struct(...
            'mode', M, ...
            'inner_iterations', n_iter, ...
            'omega_M', omega_M, ...
            'reconstruction_error', reconstruction_error, ...
            'outer_metric', outer_metric)];
        
        if params.verbose
            freq_hz = omega_M / (2*pi);  % Normalized frequency
            fprintf('  Mode %d: ω_M = %.4f rad (f = %.4f normalized)\n', M, omega_M, freq_hz);
            fprintf('  Reconstruction error: %.6f\n', reconstruction_error);
            fprintf('  Outer metric: |α - MSE|/α = %.4f (κ₂ = %.4f)\n', outer_metric, params.kappa2);
        end
        
        if outer_metric < params.kappa2
            if params.verbose
                fprintf('\nOuter loop converged after M = %d modes.\n', M);
            end
            break;
        end
    end
    
    %% Final residual
    residual = signal - sum(modes, 1)';
    
    %% Output info
    info = struct();
    info.num_modes = M;
    info.convergence_history = convergence_history;
    info.final_reconstruction_error = norm(residual)^2 / N;
    info.x_u = x_u;
    info.params = params;
    
    if params.verbose
        fprintf('\n========================================\n');
        fprintf('DMD Complete\n');
        fprintf('Extracted M = %d modes\n', M);
        fprintf('Final reconstruction error: %.6f\n', info.final_reconstruction_error);
        fprintf('Output: U = {u_k}_{k=1}^%d, W = {ω_k}_{k=1}^%d\n', M, M);
        fprintf('========================================\n');
    end
end

%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function params = set_default(params, field, value)
    % Set default parameter value if not specified
    if ~isfield(params, field)
        params.(field) = value;
    end
end

function [U_M, omega_M] = initialize_mode(signal, modes, N, omega)
    % Initialize new mode from residual signal
    %
    % Strategy: Find dominant frequency in residual spectrum
    
    % Compute residual
    if isempty(modes)
        residual = signal;
    else
        residual = signal - sum(modes, 1)';
    end
    
    % FFT of residual
    R = fft(residual);
    
    % Find dominant frequency in positive spectrum
    R_positive = abs(R(1:floor(N/2)+1));
    [~, idx] = max(R_positive);
    omega_M = omega(idx);
    
    % Initialize U_M as bandpass filtered version of residual
    % Centered around dominant frequency
    bandwidth = 0.5;  % Initial bandwidth parameter
    weight = exp(-((omega - omega_M).^2) / (2 * bandwidth^2));
    U_M = R .* weight;
end

%% ========================================================================
%  ADDITIONAL UTILITY FUNCTIONS
%% ========================================================================

function [amplitude, frequency] = compute_instantaneous_params(mode, fs)
    % Compute instantaneous amplitude and frequency using Hilbert transform
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    if nargin < 2
        fs = 1;
    end
    
    mode = mode(:);
    N = length(mode);
    
    % Hilbert transform via FFT (Eq. 6-8)
    Mode_fft = fft(mode);
    H = zeros(N, 1);
    if mod(N, 2) == 0
        H(2:N/2) = -1j;
        H(N/2+2:N) = 1j;
    else
        H(2:ceil(N/2)) = -1j;
        H(ceil(N/2)+1:N) = 1j;
    end
    
    hilbert_mode = real(ifft(Mode_fft .* H));
    
    % Analytic signal (Eq. 8)
    z = mode + 1j * hilbert_mode;
    
    % Instantaneous amplitude: A[n] = |z[n]}
    amplitude = abs(z);
    
    % Instantaneous frequency: ω[n] = φ[n+1] - φ[n]
    phase = unwrap(angle(z));
    frequency = [diff(phase); phase(end) - phase(end-1)] * fs / (2*pi);
end

function energy = compute_mode_energy(mode)
    % Compute energy of a mode: ||u||²₂
    energy = sum(abs(mode).^2);
end

function reconstructed = reconstruct_signal(modes, x_u)
    % Reconstruct signal from modes and unprocessed component
    % x[n] = Σ u_k[n] + x_u[n]
    
    if nargin < 2
        x_u = 0;
    end
    
    reconstructed = sum(modes, 1)' + x_u(:);
end
