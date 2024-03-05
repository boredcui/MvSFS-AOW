function parents = tournamentSelection(population, fitness, tournamentSize)
    % population: 种群，每行是一个个体
    % fitness: 每个个体的适应度值
    % tournamentSize: 锦标赛规模，即每次选择比赛的个体数量
    
    numParents = size(population, 1);
    
    % 初始化选出的父代
    parents = zeros(numParents, size(population, 2));
    
    for i = 1:numParents
        % 随机选择锦标赛规模个个体的索引
        tournamentIndices = randperm(numParents, tournamentSize);
        
        % 从锦标赛中选择适应度最好的个体
        [~, winnerIndex] = max(fitness(tournamentIndices));
        winner = population(tournamentIndices(winnerIndex), :);
        
        % 将锦标赛的胜者添加到父代中
        parents(i, :) = winner;
    end
end
