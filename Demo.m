%% DEMO FILE
clc;clear;close all
% Include dependencies
addpath('./lib');
addpath('./data/');

%% Load the data and select features for classification
load('MSRCV1.mat');
m = 40;
%% stable feature selection
SX = MvSFS(X, Y, m);
%% Define the parameter search ranges
% w_values = [0.9, 1.2, 1.5, 1.8];
% c1_c2_combinations = [2, 2; 1.6, 1.8; 1.6, 2];
w_values = [0.9];
c1_c2_combinations = [2, 2];

% Initialize a cell array to store results for each combination
all_results = cell(length(w_values), size(c1_c2_combinations, 1));

% Loop through parameter combinations
for w_idx = 1:length(w_values)
    w = w_values(w_idx);
    
    for c1_c2_idx = 1:size(c1_c2_combinations, 1)
        c1 = c1_c2_combinations(c1_c2_idx, 1);
        c2 = c1_c2_combinations(c1_c2_idx, 2);
        
        % Call the AOW function with current parameter values
        [final_feature_selection, particles, global_positions, global_fitness,global_std, num_iter,time] = AOW(SX, Y, m, w, c1, c2);
        
        % Save all returned values in a struct for the current combination
        result_struct = struct('final_feature_selection', final_feature_selection, ...
            'particles', particles, 'global_positions', global_positions, ...
            'global_fitness', global_fitness, 'global_std', global_std,'num_iterations', num_iter,'times', time);
        
        % Store the result struct in the cell array
        all_results{w_idx, c1_c2_idx} = result_struct;
    end
end

%% 找到准确度最高的结果
% Initialize variables to track the best result
best_global_fitness = -inf;
best_combination = struct('w', 0, 'c1', 0, 'c2', 0);
best_final_feature_selection = [];
best_global_fitness_curve = [];
best_num_iterations = 0;

% Iterate through all parameter combinations
for w_idx = 1:length(w_values)
    w = w_values(w_idx);
    
    for c1_c2_idx = 1:size(c1_c2_combinations, 1)
        c1 = c1_c2_combinations(c1_c2_idx, 1);
        c2 = c1_c2_combinations(c1_c2_idx, 2);
        
        % Access the results for the current combination
        result_struct = all_results{w_idx, c1_c2_idx};
        global_fitness_curve = result_struct.global_fitness;
        num_iterations = result_struct.num_iterations;
        
        % Check if the last value of global_fitness is the best so far
        if global_fitness_curve(end) > best_global_fitness
            best_global_fitness = global_fitness_curve(end);
            best_combination.w = w;
            best_combination.c1 = c1;
            best_combination.c2 = c2;
            best_final_feature_selection = result_struct.final_feature_selection;
            best_global_fitness_curve = global_fitness_curve;
            best_num_iterations = num_iterations;
            best_result_struct = result_struct;
        end
    end
end