clear
clc

%% a
load data
Num=unique(Table(2:end,1),'stable');
T=[]; % 记录复查时间
ED=[]; % 记录水肿数据
A=[]; % 记录除此诊断信息
for i=1:length(Num) % 循环遍历每个病人
    a=find(Table(:,1)==Num(i)); % 找到该病人的所有数据
    A=[A;double(Table(a(1),4:end))]; % 记录该病人的除此诊断信息
    T=[T;double(Table(a,3))]; % 记录该病人的复查时间
    ED=[ED;double(Table(a,34))]; % 记录该病人的水肿数据
end
figure % 新建一个图形窗口
plot(T,ED,'*') % 绘制散点图
xlim([1,2000])
% 高斯模型
hold on % 保持当前图形窗口，并在其上绘制新图形
gaussModel = fit(T, ED, 'gauss1') % 拟合高斯模型
plot(gaussModel) % 绘制高斯模型曲线
xlabel('时间')
ylabel('水肿/10^-3ml')
% 计算残差
ED_fit=gaussModel(T); % 计算拟合值
Error=[]; % 记录残差
for i=1:length(Num) % 循环遍历每个病人
    a=find(Table(:,1)==Num(i))-1; % 找到该病人的所有数据的前一行
    Error(i,1)=mean(abs(ED_fit(a)-ED(a))); % 计算残差
end

%% b
% 个体差异用Tabel第4-15列的指标来进行聚类
cluster_n=5; % 聚类中心
[center, U, obj_fcn] = fcm(double(A(:,1:12)), cluster_n); % 模糊C均值聚类
figure
plot(obj_fcn)
xlabel('iteration')
ylabel('obj.fcn_value')
title('FCM聚类')
[~,u]=max(U); % 所属亚类
ED_fit2=[]; % 记录拟合值
for i=1:cluster_n % 循环遍历每个亚类
    a=find(u==i); % 找到属于该亚类的病人编号
    b=find(ismember(Table(2:end,1),Num(a))==1); % 找到属于该亚类的数据
    figure
    plot(T(b),ED(b),'*')
    xlim([1,2000])
    % 高斯模型
    hold on
    disp('亚类1：')
    gaussModel = fit(T(b), ED(b), 'gauss1') % 拟合高斯模型
    plot(gaussModel) % 绘制高斯模型曲线
    xlabel('时间')
    ylabel('水肿/10^-3ml')
    title(['亚类',num2str(i)])
    ED_fit2(b,1)=gaussModel(T(b)); % 计算拟合值
end
% 计算残差
Error2=[];
for i=1:length(Num) % 循环遍历每个病人
    a=find(Table(:,1)==Num(i))-1; % 找到该病人的所有数据的前一行
    Error2(i,1)=mean(abs(ED_fit2(a)-ED(a))); % 计算残差
end
result2=[["患者","残差(全体)","残差(亚类)","亚类"];[Num,Error,Error2,u']];

%% c
% 计算水肿指标的变化率，在不同治疗方法下，改善效率
K=[]; % 初始化变化率
for i=1:length(Num) % 循环遍历每个病人
    a=find(Table(:,1)==Num(i)); % 找到该病人的所有数据
    if length(a)>=5 % 至少五次检查数据才进行计算
        k=(double(Table(a(2:end),34))-double(Table(a(1:end-1),34)))./(double(Table(a(2:end),3))-double(Table(a(1:end-1),3))); % 计算变化率
        kk=find(k<0); % 找到变化率小于0的数据
        if length(kk)==0
            K(i,1)=0; % 如果没有变化率小于0的数据，则变化率为0
        else
            K(i,1)=mean(k(kk)); % 计算变化率的平均值
        end
    else
        K(i,1)=NaN; % 如果数据不足五次，则变化率为NaN
    end
end
c=setdiff([1:length(K)],find(isnan(K)==1)); % 找到变化率不为NaN的数据的索引
G=double(Table(2:end,16:22)); % 获取治疗方法数据
Z=Table(1,16:22); % 获取治疗方法名称
for i=1:length(Z) % 循环遍历每种治疗方法
    [p(i),anovatab{i},stats{i}]=anova1(K(c),G(c,i)+1,'off'); % 单因素方差分析
    fa=finv(0.95,anovatab{i}{2,3},anovatab{i}{3,3}); % 计算fa
    F=anovatab{i}{2,5}; % F值
    if p(i)<=0.01 && F>fa
        disp([Z(i)+"对水肿体积进展模式影响非常显著"])
        fprintf('p值为%.4f<0.01，F值为%.2f>%.2f\n',p(i),F,fa)
    elseif p(i)<=0.05 && F>fa
        disp([Z(i)+"对水肿体积进展模式影响显著"])
        fprintf('p值为%.4f<0.05，F值为%.2f>%.2f\n',p(i),F,fa)
    else
        disp([Z(i)+"对水肿体积进展模式影响不显著"])
        fprintf('p值为%.4f，F值为%.2f\n',p(i),F)
    end
end
disp('不同治疗对水肿进展模式的影响大小为：')
[~,q]=sort(p);
str=Z(q)+"">"";
disp(str)

%% d
% 同上述步骤求血肿体积的
% 计算水肿指标的变化率，在不同治疗方法下，改善效率
K2=[]; % 初始化变化率
for i=1:length(Num) % 循环遍历每个病人
    a=find(Table(:,1)==Num(i)); % 找到该病人的所有数据
    if length(a)>=5 % 至少五次检查数据才进行计算
        k=(double(Table(a(2:end),23))-double(Table(a(1:end-1),23)))./(double(Table(a(2:end),3))-double(Table(a(1:end-1),3))); % 计算变化率
        kk=find(k<0); % 找到变化率小于0的数据
        if length(kk)==0
            K2(i,1)=0; % 如果没有变化率小于0的数据，则变化率为0
        else
            K2(i,1)=mean(k(kk)); % 计算变化率的平均值
        end
    else
        K2(i,1)=NaN; % 如果数据不足五次，则变化率为NaN
    end
end
c=setdiff([1:length(K2)],find(isnan(K2)==1)); % 找到变化率不为NaN的数据的索引
for i=1:length(Z) % 循环遍历每种治疗方法
    [p2(i),anovatab2{i},stats2{i}]=anova1(K2(c),G(c,i)+1,'off'); % 单因素方差分析
    fa=finv(0.95,anovatab2{i}{2,3},anovatab2{i}{3,3}); % 计算fa
    F=anovatab2{i}{2,5}; % F值
    if p2(i)<=0.01 && F>fa
        disp([Z(i)+"对血肿肿体积进展模式影响非常显著"])
        fprintf('p值为%.4f<0.01，F值为%.2f>%.2f\n',p2(i),F,fa)
    elseif p2(i)<=0.05 && F>fa
        disp([Z(i)+"对血肿体积进展模式影响显著"])
        fprintf('p值为%.4f<0.05，F值为%.2f>%.2f\n',p2(i),F,fa)
    else
        disp([Z(i)+"对血肿体积进展模式影响不显著"])
        fprintf('p值为%.4f，F值为%.2f\n',p2(i),F)
    end
end
disp('不同治疗对血肿进展模式的影响大小为：')
[~,q2]=sort(p2);
str2=Z(q2)+"">"";
disp(str2)
% 计算血肿指标与水肿指标的相关性
x0=double(Table(2:end,23)); % 获取血肿数据
y0=double(Table(2:end,34)); % 获取水肿数据
theta=x0'*y0/(norm(x0)*norm(y0)); % 余弦相似度
fprintf('血肿指标与水肿指标的相关度为：%.4f\n',theta)

disp('结果见矩阵result2')