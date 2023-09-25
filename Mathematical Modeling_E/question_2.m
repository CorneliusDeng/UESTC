clear
clc

%% a
load data
Num=unique(Table(2:end,1),'stable');
T=[];%记录复查时间
ED=[];%记录水肿数据
A=[];%记录除此诊断信息
for i=1:length(Num)
    a=find(Table(:,1)==Num(i));
    A=[A;double(Table(a(1),4:end))];
    T=[T;double(Table(a,3))];
    ED=[ED;double(Table(a,34))];
end
figure
plot(T,ED,'*')
xlim([1,2000])
%高斯模型
hold on
gaussModel = fit(T, ED, 'gauss1')
plot(gaussModel)
xlabel('时间')
ylabel('水肿/10^-3ml')
%计算残差
ED_fit=gaussModel(T);
Error=[];
for i=1:length(Num)
    a=find(Table(:,1)==Num(i))-1;
    Error(i,1)=mean(abs(ED_fit(a)-ED(a)));
end
%% b
%个体差异用Tabel第4-15列的指标来进行聚类
cluster_n=5;%聚类中心
[center, U, obj_fcn] = fcm(double(A(:,1:12)), cluster_n);
figure%目标函数变化值
plot(obj_fcn)
xlabel('iteration')
ylabel('obj.fcn_value')
title('FCM聚类')
[~,u]=max(U);%所属亚类
ED_fit2=[];
for i=1:cluster_n
    a=find(u==i);
    b=find(ismember(Table(2:end,1),Num(a))==1);
    figure
    plot(T(b),ED(b),'*')
    xlim([1,2000])
    %高斯模型
    hold on
    disp('亚类1：')
    gaussModel = fit(T(b), ED(b), 'gauss1')
    plot(gaussModel)
    xlabel('时间')
    ylabel('水肿/10^-3ml')
    title(['亚类',num2str(i)])
    ED_fit2(b,1)=gaussModel(T(b));
end
%计算残差
Error2=[];
for i=1:length(Num)
    a=find(Table(:,1)==Num(i))-1;
    Error2(i,1)=mean(abs(ED_fit2(a)-ED(a)));
end
result2=[["患者","残差(全体)","残差(亚类)","亚类"];[Num,Error,Error2,u']];
%% c
K = [];
for i = 1:length(Num)
    a = find(Table(:, 1) == Num(i));
    if length(a) >= 5 % 至少五次检查数据才进行计算
        k = (double(Table(a(2:end), 34)) - double(Table(a(1:end-1), 34))) ./ (double(Table(a(2:end), 3)) - double(Table(a(1:end-1), 3)));
        kk = find(k < 0);
        if isempty(kk)
            K(i, 1) = 0;
        else
            K(i, 1) = mean(k(kk));
        end
    else
        K(i, 1) = NaN;
    end
end

c = setdiff([1:length(K)], find(isnan(K) == 1));
G = double(Table(2:end, 16:22));
Z = Table(1, 16:22);

mdl = fitlm(G(c, :), K(c)); % 多元线性回归分析

disp(mdl)

disp('不同治疗对水肿进展模式的影响大小为：')
[~, q] = sort(mdl.Coefficients.pValue(2:end));
str = Z(q) + " < ";
disp(str)

disp('每个变量对应的p值为：')
pValues = mdl.Coefficients.pValue(q+1);
variableNames = Z(q);

for i = 1:length(variableNames)
    disp([variableNames{i} + ": " + num2str(pValues(i))])
end
disp('========================================================')
%% d
% 计算血肿体积的变化率
% 初始化变量
K2 = [];
for i = 1:length(Num)
    a = find(Table(:,1) == Num(i));
    if length(a) >= 5 % 至少五次检查数据才进行计算
        k = (double(Table(a(2:end), 23)) - double(Table(a(1:end-1), 23))) ./ (double(Table(a(2:end), 3)) - double(Table(a(1:end-1), 3)));
        kk = find(k < 0);
        if isempty(kk)
            K2(i,1) = 0;
        else
            K2(i,1) = mean(k(kk));
        end
    else
        K2(i,1) = NaN;
    end
end

c = setdiff([1:length(K2)], find(isnan(K2) == 1));
K2 = K2(c);
G = G(c,:);

% 多元线性回归分析
mdl = fitlm(G, K2);

% 显示结果
disp(mdl)

Z = Table(1, 16:22);
disp('不同治疗对水肿进展模式的影响大小为：')
[~, q] = sort(mdl.Coefficients.pValue(2:end));
str = Z(q) + " < ";
disp(str)

disp('每个变量对应的p值为：')
pValues = mdl.Coefficients.pValue(q+1);
variableNames = Z(q);

% 计算血肿指标与水肿指标的相关性
x0 = double(Table(2:end, 23));
y0 = double(Table(2:end, 34));
theta = x0' * y0 / (norm(x0) * norm(y0)); % 余弦相似度
fprintf('血肿指标与水肿指标的相关度为：%.4f\n', theta)

disp('结果见矩阵result2')