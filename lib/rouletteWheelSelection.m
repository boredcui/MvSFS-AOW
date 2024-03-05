function parents = rouletteWheelSelection(population, fitness)
    % 根据适应度值选择父母个体
    % population 是当前种群，fitness 是适应度值
    
    % 计算适应度值的总和
    totalFitness = sum(fitness);
    
    % 生成一个轮盘赌选择概率分布
    selectionProbabilities = fitness / totalFitness;
    
    % 使用轮盘赌法选择父母
    numParents = size(population, 1);
    selectedParents = zeros(numParents, size(population, 2));
    
    for i = 1:numParents
        % 在概率分布中选择一个位置
        rouletteSpin = rand();
        
        % 选择父母个体
        cumulativeProbability = 0;
        for j = 1:numParents
            cumulativeProbability = cumulativeProbability + selectionProbabilities(j);
            if cumulativeProbability >= rouletteSpin
                selectedParents(i, :) = population(j, :);
                break;
            end
        end
    end
    
    parents = selectedParents;
end

