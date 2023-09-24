clear
clc

load data
Num = unique(Table(2:end,1),'stable');
A = []; % 首检
Y = []; % 首检Rms
B = []; % 随检
N1 = [];
N2 = [];
Z = [Table(1,[4:end])];
for i = 1:length(Num)
    a = find(Table(:,1) == Num(i));
    A = [A; double(Table(a(1),[4:end]))];
    N1 = [N1; Table(a(1),1:2)];
    B = [B; double(Table(a(2:end),[4:end]))];
    N2 = [N2; Table(a(2:end),1:2)];
end

%% a
In = A;
Y = xlsread('表1-患者列表及临床信息.xlsx','B2:B101');
Out = Y;

% 数据降维（PCA）
numComponents = 10; % 设置降维后的维数
[coeff, score, ~, ~, explained] = pca(In);
In_pca = score(:, 1:numComponents);

Xtrain = In_pca(1:100,:);
Ytrain = Out;
Xtest1 = In_pca(101:130,:);
Xtest2 = In_pca(131:160,:);

%%  训练模型
trees = 100;                                      % 决策树数目
leaf  = 5;                                        % 最小叶子数
OOBPrediction = 'on';                             % 打开误差图
OOBPredictorImportance = 'on';                    % 计算特征重要性
Method = 'classification';                            % 分类还是回归
net = TreeBagger(trees, Xtrain, Ytrain, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);

% 预测
Y = string(predict(net,Xtrain));
Y1 = string(predict(net,Xtest1));
Y2 = string(predict(net,Xtest2));

% 混淆矩阵
figure
plotconfusion(categorical(Ytrain), categorical(Y));
title('训练集')
figure
hold on
auc = plot_roc(double(Y)', Ytrain');
legend(['训练集 auc=',num2str(round(auc,2))])

%% b
Xyuce = B;
In_pca_yuce = (Xyuce - mean(In)) * coeff(:,1:numComponents); % 在预测集上应用相同的PCA转换
Y3 = string(predict(net, In_pca_yuce));

result3 = "";
for i = 1:length(Num)
    a = find(N2 == Num(i));
    for j = 1:length(a)
        result3(i,j) = Y3(a(j));
    end
end
result3(ismissing(result3) == 1) = "";
z = [];
for i = 1:size(result3,2)
    z = [z,"第"+num2str(i)+"次随访"];
end
result3 = [["患者","首次检测";Num,[Y;Y1;Y2]],[z;result3]];

%% c
% 相关性分析
X = double(Table(2:end,[4:end]));
P = corrcoef(X);

figure % 热图
set(gca,'position',[0.05 0.05 0.9 0.9])
h = heatmap(P,'ColorbarVisible', 'on');
h.FontSize = 8;
h.CellLabelFormat = '%0.2g';

resultp = ["",Z;Z',P]; % 自行分析相关性结果

disp('预测结果见result3，相关性结果见resultp')