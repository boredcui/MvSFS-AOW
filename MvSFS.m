function SX = MvSFS(X, Y, m)

addpath('./data/');
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

sz = size(X);
v = sz(1);
numF = m;

for i = 1:v
    X{i} = normalizemeanstd(X{i});
    % mrmr
    index_mrmr{i} = mRMR(X{i}, Y, numF);
    fel_mrmr{i} = X{i}(:,index_mrmr{i}(:,1:end));
    [acc_mrmr] = SVM(fel_mrmr{i},Y);
    
    % mcfs
    options = [];
    options.gnd = Y;
    [FeaIndex] = MCFS_p(X{i}, numF,options);
    index_mcfs{i} = FeaIndex{1}';
    fel_mcfs{i} = X{i}(:,index_mcfs{i}(:,1:end));
    [acc_mcfs] = SVM(fel_mcfs{i},Y);
    
    % cfs
    rank_cfs = cfs(X{i});
    rank_cfs = rank_cfs';
    index_cfs{i} = rank_cfs(:,1:numF);
    fel_cfs{i} = X{i}(:,index_cfs{i}(:,1:end));
    [acc_cfs] = SVM(fel_cfs{i},Y);
    
    % reliefF
    rank_reliefF = reliefF( X{i}, Y, 20);
    index_reliefF{i} = rank_reliefF(:,1:numF);
    fel_reliefF{i} = X{i}(:,index_reliefF{i}(:,1:end));
    [acc_reliefF] = SVM(fel_reliefF{i},Y);
    
    % calculate method weight
    mrmr_weight(i) = acc_mrmr/(acc_mrmr+acc_mcfs+acc_cfs+acc_reliefF);
    mcfs_weghit(i) = acc_mcfs/(acc_mrmr+acc_mcfs+acc_cfs+acc_reliefF);
    cfs_weghit(i) = acc_cfs/(acc_mrmr+acc_mcfs+acc_cfs+acc_reliefF);
    reliefF_weghit(i) = acc_reliefF/(acc_mrmr+acc_mcfs+acc_cfs+acc_reliefF);

    % calculate feature weight
    for j = 1:numF
        fel_mrmr_weight{i}(j) = mrmr_weight(i)*(numF-j+1)/numF;
        fel_mcfs_weight{i}(j) = mcfs_weghit(i)*(numF-j+1)/numF;
        fel_cfs_weight{i}(j) = cfs_weghit(i)*(numF-j+1)/numF;
        fel_reliefF_weight{i}(j) = reliefF_weghit(i)*(numF-j+1)/numF;
    end

    combinedIndex{i} = [index_mrmr{i}, index_mcfs{i}, index_cfs{i}, index_reliefF{i}];
    combinedWeight{i} = zeros(size(combinedIndex{i}));
    
    for k = 1:length(combinedIndex{i})
        idx = combinedIndex{i}(k);
        combinedWeight{i}(k) = sum(fel_mrmr_weight{i}(index_mrmr{i} == idx)) + sum(fel_mcfs_weight{i}(index_mcfs{i} == idx)) ...
        +sum(fel_cfs_weight{i}(index_cfs{i} == idx))+sum(fel_reliefF_weight{i}(index_reliefF{i} == idx));
    end
    
    [init_index, uniqueIdx] = unique(combinedIndex{i});
    fea_weight{i} = combinedWeight{i}(uniqueIdx);
    [sorted_weights, sorted_indices] = sort(fea_weight{i}, 'descend');
    sorted_init_index{i} = init_index(sorted_indices);
    % 使用排好序的索引从特征矩阵中提取相应的特征
    sorted_init_fea{i} = X{i}(: ,sorted_init_index{i}(:,1:numF));
end
SX = sorted_init_fea';