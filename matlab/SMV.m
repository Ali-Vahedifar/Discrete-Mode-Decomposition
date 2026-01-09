%% Shapley Mode Value (SMV) - Monte Carlo Approximation
%  =====================================================
%  
%  Implementation of the Shapley Mode Value algorithm as described in
%  Section V and Algorithm 2 of the IEEE INFOCOM 2025 paper.
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
%  The Shapley Mode Value X_i satisfies two axioms:
%  1. Transferability: Σ X_i = V(D) (sum equals total utility)
%  2. Monotonicity: Combines null contribution, symmetry, and linearity
%
%  The unique solution is (Eq. 33):
%    X_i = Σ_{S⊆D\{i}} [|S|!(|D|-|S|-1)!/|D|!] * [V(S∪{i}) - V(S)]
%
%  Monte Carlo approximation reduces complexity from O(2^M) to O(P·M),
%  where P is the number of permutations.
%
%  Convergence criterion (Eq. 36):
%    (1/M) Σ |X_i^t - X_i^{t-100}| / |X_i^t| < 0.01
%
%  Usage:
%    [shapley_values, rankings] = SMV(modes, center_freqs, predictor, X_val, y_val)
%    [shapley_values, rankings, info] = SMV(modes, center_freqs, predictor, X_val, y_val, params)
%
%  Inputs:
%    modes        - Extracted modes (M x N matrix)
%    center_freqs - Center frequencies (M x 1 vector)
%    predictor    - Function handle: predictions = predictor(selected_modes, X)
%    X_val        - Validation input data
%    y_val        - Validation target data
%    params       - (Optional) Algorithm parameters
%
%  Outputs:
%    shapley_values - Shapley values for each mode (M x 1)
%    rankings       - Mode indices sorted by importance (descending)
%    info           - Convergence and algorithm information

function [shapley_values, rankings, info] = SMV(modes, center_freqs, predictor, X_val, y_val, params)
    
    %% Default parameters
    if nargin < 6
        params = struct();
    end
    
    params = set_default(params, 'tolerance', 0.01);          % Convergence tolerance (1%)
    params = set_default(params, 'max_iterations', 1000);     % Max Monte Carlo iterations
    params = set_default(params, 'epsilon3', 0.01);           % Performance tolerance
    params = set_default(params, 'convergence_window', 100);  % Window for convergence check
    params = set_default(params, 'metric', 'accuracy');       % Performance metric
    params = set_default(params, 'verbose', true);            % Print progress
    params = set_default(params, 'seed', 42);                 % Random seed
    
    %% Initialize
    rng(params.seed);  % For reproducibility
    
    M = size(modes, 1);  % Number of modes
    
    if M == 0
        shapley_values = [];
        rankings = [];
        info = struct('num_iterations', 0, 'converged', true);
        return;
    end
    
    % Initialize Shapley values
    shapley_values = zeros(M, 1);
    shapley_history = zeros(M, params.max_iterations);
    
    % Compute V(D) - performance with all modes
    V_D = evaluate_coalition(modes, 1:M, predictor, X_val, y_val, params.metric);
    
    % Compute V(∅) - baseline performance
    V_empty = evaluate_coalition(modes, [], predictor, X_val, y_val, params.metric);
    
    if params.verbose
        fprintf('========================================\n');
        fprintf('Shapley Mode Value (SMV) Computation\n');
        fprintf('Author: Ali Vahedi (Mohammad Ali Vahedifar)\n');
        fprintf('IEEE INFOCOM 2025\n');
        fprintf('========================================\n');
        fprintf('Number of modes: %d\n', M);
        fprintf('V(D) = %.4f, V(∅) = %.4f\n', V_D, V_empty);
        fprintf('----------------------------------------\n');
    end
    
    %% Monte Carlo Shapley Value Approximation (Algorithm 2)
    converged = false;
    t = 0;
    
    while ~converged && t < params.max_iterations
        t = t + 1;
        
        % Random permutation of modes
        perm = randperm(M);
        
        % Initialize performance for empty set
        v_prev = V_empty;
        
        % Scan through permutation
        for j = 1:M
            mode_idx = perm(j);
            
            % Current coalition (modes before j in permutation)
            coalition = perm(1:j);
            
            % Early stopping: if close to V(D), skip computation
            if abs(V_D - v_prev) < params.epsilon3
                v_curr = v_prev;
            else
                % Compute V(S ∪ {i})
                v_curr = evaluate_coalition(modes, coalition, predictor, X_val, y_val, params.metric);
            end
            
            % Update Shapley value using running average
            % X_i = (t-1)/t * X_i^{t-1} + 1/t * (v_j - v_{j-1})
            marginal_contribution = v_curr - v_prev;
            shapley_values(mode_idx) = ((t-1) * shapley_values(mode_idx) + marginal_contribution) / t;
            
            v_prev = v_curr;
        end
        
        % Store history
        shapley_history(:, t) = shapley_values;
        
        % Check convergence (Eq. 36)
        if t > params.convergence_window
            old_values = shapley_history(:, t - params.convergence_window);
            relative_change = mean(abs(shapley_values - old_values) ./ (abs(shapley_values) + 1e-10));
            
            if relative_change < params.tolerance
                converged = true;
                if params.verbose
                    fprintf('Converged at iteration %d (relative change: %.4f)\n', t, relative_change);
                end
            end
        end
        
        % Progress update
        if params.verbose && mod(t, 100) == 0
            fprintf('Iteration %d: mean |X| = %.4f\n', t, mean(abs(shapley_values)));
        end
    end
    
    %% Rank modes by Shapley value (descending)
    [~, rankings] = sort(shapley_values, 'descend');
    
    %% Select top K modes (modes with positive contribution)
    K = sum(shapley_values > 0);
    if K == 0
        K = max(1, floor(M/2));  % At least select half
    end
    top_K_modes = rankings(1:K);
    
    %% Output info
    info = struct();
    info.num_iterations = t;
    info.converged = converged;
    info.V_D = V_D;
    info.V_empty = V_empty;
    info.top_K_modes = top_K_modes;
    info.K = K;
    info.shapley_history = shapley_history(:, 1:t);
    
    if params.verbose
        fprintf('\n========================================\n');
        fprintf('SMV Complete\n');
        fprintf('Iterations: %d, Converged: %s\n', t, mat2str(converged));
        fprintf('Top %d modes selected\n', K);
        fprintf('Rankings: %s\n', mat2str(rankings'));
        fprintf('Shapley values: %s\n', mat2str(shapley_values', 4));
        fprintf('========================================\n');
    end
end

%% Helper Functions

function params = set_default(params, field, value)
    if ~isfield(params, field)
        params.(field) = value;
    end
end

function performance = evaluate_coalition(modes, coalition, predictor, X_val, y_val, metric)
    % Evaluate performance of a coalition of modes
    % This is the performance oracle V(S, Z) from the paper
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    if isempty(coalition)
        % Empty coalition - baseline performance
        switch metric
            case 'accuracy'
                performance = 0;
            case 'mse'
                performance = -mean((y_val - mean(y_val)).^2);  % Negative MSE
            case 'mae'
                performance = -mean(abs(y_val - mean(y_val)));
            otherwise
                performance = 0;
        end
        return;
    end
    
    % Select modes in coalition
    selected_modes = modes(coalition, :);
    
    % Get predictions
    try
        predictions = predictor(selected_modes, X_val);
        
        % Compute performance metric
        switch metric
            case 'accuracy'
                % For regression: accuracy = (1 - MAE/range) * 100
                mae = mean(abs(predictions(:) - y_val(:)));
                data_range = max(y_val(:)) - min(y_val(:)) + 1e-10;
                performance = (1 - mae / data_range) * 100;
                
            case 'mse'
                % Negative MSE (higher is better)
                performance = -mean((predictions(:) - y_val(:)).^2);
                
            case 'mae'
                % Negative MAE (higher is better)
                performance = -mean(abs(predictions(:) - y_val(:)));
                
            case 'psnr'
                % PSNR
                mse = mean((predictions(:) - y_val(:)).^2);
                if mse < 1e-10
                    performance = 100;
                else
                    max_val = max(abs(y_val(:)));
                    performance = 10 * log10(max_val^2 / mse);
                end
                
            otherwise
                error('Unknown metric: %s', metric);
        end
    catch
        performance = 0;
    end
end

function [shapley_values, rankings] = SMV_exact(modes, predictor, X_val, y_val, metric)
    % Exact Shapley value computation (for small M)
    % Complexity: O(2^M)
    %
    % Author: Ali Vahedi (Mohammad Ali Vahedifar)
    % IEEE INFOCOM 2025
    
    if nargin < 5
        metric = 'accuracy';
    end
    
    M = size(modes, 1);
    shapley_values = zeros(M, 1);
    
    % Precompute factorials
    fact = factorial(0:M);
    
    % Iterate over all subsets
    for i = 1:M
        other_indices = setdiff(1:M, i);
        
        for subset_size = 0:M-1
            % Generate all subsets of size subset_size from other_indices
            if subset_size == 0
                subsets = {[]};
            else
                subsets = nchoosek(other_indices, subset_size);
                subsets = num2cell(subsets, 2);
            end
            
            for s = 1:length(subsets)
                subset = subsets{s};
                if iscell(subset)
                    subset = subset{1};
                end
                
                % Weight: |S|!(M-|S|-1)!/M!
                S_size = length(subset);
                weight = fact(S_size + 1) * fact(M - S_size) / fact(M + 1);
                
                % V(S ∪ {i})
                V_with_i = evaluate_coalition(modes, [subset, i], predictor, X_val, y_val, metric);
                
                % V(S)
                V_without_i = evaluate_coalition(modes, subset, predictor, X_val, y_val, metric);
                
                % Update Shapley value
                shapley_values(i) = shapley_values(i) + weight * (V_with_i - V_without_i);
            end
        end
    end
    
    % Rank modes
    [~, rankings] = sort(shapley_values, 'descend');
end
