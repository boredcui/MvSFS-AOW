%% fitness
function [acc,s] = SVM(X,Y)
classcode=unique(Y);
data=X;
label=Y;  %要求数据集每一行代表一个样本 ;label每行为标签
 
% [M,N]=size(X); % M:总样本量； N:一个样本的元素总数
% indices=crossvalind('Kfold',data(1:M,N),10);  %进行随机分包
%  
% for k=1:10  %交叉验证k=10，10个包轮流作为测试集
%     test = (indices == k);   %获得test集元素在数据集中对应的单元编号
%     train = ~test;  %train集元素的编号为非test元素的编号
%     train_data=data(train,:);%从数据集中划分出train样本的数据
%     train_label=label(train,:);
%     test_data=data(test,:);  %test样本集
%     test_label=label(test,:);
%     
%     %LR分类
%     model = classf_svm_tr(train_data,train_label);
%     %LR网络预测
%     [predict_label] = classf_svm_te(model,test_data,test_label);
%     [Conf,metric]=ConfusionMatrix(predict_label,test_label,classcode);
%     Acc(k) = metric.accuracy;
% end

nbtrain=round(0.5*size(Y,1));
classcode=unique(Y);
for i=1:20 % 运行20次取平均值
    [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(X, Y, nbtrain,classcode);
    [xapp,xtest]=normalizemeanstd(xapp,xtest);
    %svm分类
    model = svmtrain(yapp,xapp);
    % SVM网络预测
    [predict_label] = svmpredict(ytest,xtest,model);
    [Conf,metric]=ConfusionMatrix(predict_label,ytest,classcode);
    Acc(i) = metric.accuracy;
end

acc = mean(Acc);
s = std(Acc);
end