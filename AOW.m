function [final_feature_selection, all_particle_positions, all_global_best_positions, all_global_best_fitness, all_global_best_std,iteration,time] = AOW(SX, Y, m, w, c1,c2)

%% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

tic; % start timer

sz = size(SX);
v = sz(1);

% 参数设置
num_particles = 50; % 粒子数量
max_iterations = 100; % 最大迭代次数
num_views = v; % 视图数量
num_features_per_view = m; % 每个视图已选择的已排序特征数量，这里假设每个视图选取m个特征
% w = 0.9; % 惯性权重
% c1 = 2; % 个体学习因子
% c2 = 2; % 社会学习因子

% 初始化粒子群
particles_position = zeros(num_particles, num_views); % 初始化粒子位置，每个视图选择的特征数量
particles_velocity = rand(num_particles, num_views); % 初始化粒子速度
personal_best = particles_position; % 个体最佳位置初始化为当前位置
global_best = []; % 全局最佳位置
consecutive_stagnation = 0; % 连续代数中最优个体未改变的计数器
stagnation_threshold = 5; % 设定连续代数未改变的阈值

% 跟踪最佳个体和全局的适应度
personal_best_fitness = zeros(1, num_particles);
global_best_fitness = -inf; % 初始全局最佳适应度设为负无穷

% Initialize variables to store the data
all_particle_positions = zeros(max_iterations+1, num_particles, num_views);
all_global_best_positions = zeros(max_iterations+1, num_views);

% 计算初始适应度并更新全局最佳位置
for i = 1:num_particles
    % 随机初始化粒子位置，确保每个视图中选择的特征数量合为m
    particles_position(i, :) = random_feature_selection(num_views, num_features_per_view, m);
    
    % 计算适应度，包括分类准确度
    [fitness,std] = compute_fitness(particles_position(i, :), num_views,SX,Y);

    % 记录个体最佳适应度
    personal_best_fitness(i) = fitness;
    
    % 更新个体最佳位置
    personal_best(i, :) = particles_position(i, :);
    
    % 更新全局最佳位置
    if isempty(global_best) || fitness > global_best_fitness
        global_best = particles_position(i, :);
        global_best_fitness = fitness;
        global_best_std = std;
    end
end

all_particle_positions(1, :, :) = particles_position;
all_global_best_positions(1, :) = global_best;
all_global_best_fitness(1) = global_best_fitness;
all_global_best_std(1) = global_best_std;

previous_global_best = global_best; % 初始化前一代的全局最佳位置

% 开始迭代
for iteration = 1:max_iterations
    for i = 1:num_particles
        % 计算速度更新
        particles_velocity(i, :) = w * particles_velocity(i, :) ...
            + c1 * rand() * (personal_best(i, :) - particles_position(i, :)) ...
            + c2 * rand() * (global_best - particles_position(i, :));
        
        % 限制速度范围，确保在合理的范围内
        particles_velocity(i, :) = min(max(particles_velocity(i, :), -1), 1);
        
        % 更新粒子位置（使用 round 函数来确保整数值）
        particles_position(i, :) = round(particles_position(i, :) + particles_velocity(i, :));
        particles_position(i, :) = max(particles_position(i, :), 0);
        
        % 随机修正粒子位置，以确保每个视图中选择的特征数量合为m
        particles_position(i, :) = random_feature_correction(particles_position(i, :), num_views, m);
        
        % 计算适应度，包括分类准确度
        [fitness,std] = compute_fitness(particles_position(i, :), num_views, SX,Y);
        
        % 更新个体最佳位置
        if fitness > personal_best_fitness(i)
            personal_best(i, :) = particles_position(i, :);
            personal_best_fitness(i) = fitness;
        end
        
        % 更新全局最佳位置
        if fitness > global_best_fitness
            global_best = particles_position(i, :);
            global_best_fitness = fitness;
            global_best_std = std;
        end
    end

    % Record data at each iteration
    all_particle_positions(iteration+1, :, :) = particles_position;
    all_global_best_positions(iteration+1, :) = global_best;
    all_global_best_fitness(iteration+1) = global_best_fitness;
    all_global_best_std(iteration+1) = global_best_std;
    
    % 判断是否连续代数中最优个体未改变
    if iteration > 1 && isequal(global_best, previous_global_best)
        consecutive_stagnation = consecutive_stagnation + 1;
    else
        consecutive_stagnation = 0;
    end
    
    % 更新前一代的全局最佳位置
    previous_global_best = global_best;
    
    % 如果连续代数中最优个体未改变超过阈值，终止算法
    if consecutive_stagnation >= stagnation_threshold
        [fitness,std,final_feature_selection] = compute_fitness(global_best, num_views, SX,Y);
        break;
    end
end


% 计算适应度函数
function [fitness,std,feature_selection] = compute_fitness(particle, num_views, SX,Y)
    % 解析粒子中的特征选择信息
    feature_selection = [];
    for i = 1:num_views
        feature_selection = [feature_selection,SX{i}(:,1:particle(i))]
    end
    
    % 根据特征选择计算适应度，可以使用分类准确度作为适应度
    [fitness,std] = SVM(feature_selection,Y);
end

% 随机生成每个视图中选择的特征数量，确保其合为m
function feature_selection = random_feature_selection(num_views, num_features_per_view, m)
    feature_selection = zeros(1, num_views);
    
    for v = 1:num_views
        if v == num_views
            feature_selection(v) = m - sum(feature_selection);
        else
            max_features = min(num_features_per_view, m - sum(feature_selection));
            feature_selection(v) = randi([0, max_features]);
        end
    end
end

% 随机修正粒子位置，以确保每个视图中选择的特征数量合为m
function feature_selection = random_feature_correction(particle, num_views, m)
    current_selection = particle; % 当前的特征选择
    excess = sum(current_selection) - m;
    if excess > 0
        num_selected = sum(current_selection);
        while num_selected > m
            % 随机选择已选择的特征，直到特征数量合为m
            selected_features = find(current_selection);
            selected = randsample(selected_features, 1);
            current_selection(selected) = current_selection(selected)-1;
            num_selected = num_selected - 1;
        end
    elseif excess < 0
        num_selected = sum(current_selection);
        while num_selected < m
            % 随机选择已选择的特征，直到特征数量合为m
            selected = randsample(num_views, 1);
            current_selection(selected) = current_selection(selected)+1;
            num_selected = num_selected + 1;
        end
    end
    feature_selection = current_selection;
end

time = toc; % stop timer and calculate elapsed time

end

