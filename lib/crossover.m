function offspring = crossover(parents, m, crossoverRate)
    % parents: 一个矩阵，包含已经通过锦标赛选择选出的父代，每行是一个父体
    % m: 每个子代包含的特征数量
    % crossoverRate: 交叉概率，通常是一个介于 0 和 1 之间的小数
    
    numParents = size(parents, 1);
    d = size(parents, 2);
    
    % 初始化子代矩阵，数量与父代相同
    offspring = zeros(numParents, d);
    
    for i = 1:2:numParents
        % 随机选择两个不同的父代
        parentIndices = randperm(numParents, 2);
        parent1 = parents(parentIndices(1), :);
        parent2 = parents(parentIndices(2), :);
        
        % 根据交叉概率决定是否执行交叉操作
        if rand() <= crossoverRate
            % 找出两个父代共同的“优势基因位”
            commonGenes = and(parent1, parent2);

            % 找出两个父代中至少一个为1的“非优势基因位”
            nonCommonGenes = xor(parent1, parent2);

            % 计算“非优势基因位”数量
            numNonCommonGenes = sum(nonCommonGenes);

            % 计算需要引入随机性的“非优势基因位”数量
            numRandomGenes = max(0, m - sum(commonGenes));

            % 随机选择 numRandomGenes 个“非优势基因位”并将它们设置为1
            if numRandomGenes > 0
                randomGenesIndices = find(nonCommonGenes);
                randomGenesIndices = randomGenesIndices(randperm(numNonCommonGenes, numRandomGenes));
                % 在子代中设置相同的基因
                offspring(i, randomGenesIndices) = 1;
                randomGenesIndices = find(nonCommonGenes);
                randomGenesIndices = randomGenesIndices(randperm(numNonCommonGenes, numRandomGenes));
                offspring(i+1, randomGenesIndices) = 1;
            end

            % 设置共同的“优势基因位”
            offspring(i, commonGenes) = 1;
            offspring(i+1, commonGenes) = 1;
        else
            % 如果不执行交叉操作，直接复制父代到子代
            offspring(i, :) = parent1;
            offspring(i+1, :) = parent2;
        end
    end
end


