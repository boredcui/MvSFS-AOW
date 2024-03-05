function offspring = mutate(subpopulation, mutationRate)
    % subpopulation: 子代矩阵，每行是一个子代
    % mutationRate: 变异率，表示每个位上进行变异的概率
    
    numOffspring = size(subpopulation, 1);
    d = size(subpopulation, 2);
    
    % 创建一个随机的掩码矩阵，用于确定哪些位进行变异
    mutationMask = rand(numOffspring, d) < mutationRate;
    
    % 在掩码矩阵中的1位置上进行变异
    mutatedOffspring = subpopulation;
    for i = 1:numOffspring
        for j = 1:d
            if mutationMask(i, j) == 1
                % 随机选择一个与当前位不同的基因位
                randomIndex = randi([1, d]);
                while randomIndex == j
                    randomIndex = randi([1, d]);
                end
                % 交换基因位
                temp = mutatedOffspring(i, j);
                mutatedOffspring(i, j) = mutatedOffspring(i, randomIndex);
                mutatedOffspring(i, randomIndex) = temp;
            end
        end
    end
    
    offspring = mutatedOffspring;
end


