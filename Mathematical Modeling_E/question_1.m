clear
clc
warning off
load data

%% a
Num=unique(Table(2:end,1),'stable'); % Num记录患者编号，删除重复值
A=[]; % 患者第一次诊断时的指标
B=[]; % 记录拟合参数
Y=[]; % 是否发病
T=zeros(length(Num),1); % 静脉扩张时间
for i=1:length(Num) % 遍历患者
    a=find(Table(:,1)==Num(i)); % 找到患者数据
    A=[A;double(Table(a(1),4:end))]; % 存储患者第一次诊断时的指标
    t=double(Table(a,3)); % 记录时间
    HM_volume=double(Table(a,23)); % 记录血肿体积
    if length(HM_volume>0) % 如果该患者有血肿体积数据
        % b=regress(HM_volume,[ones(length(t),1),t]); % 使用线性回归算法拟合血肿体积和时间的关系，并将拟合参数存储到b中
        mdl = fitrsvm(t,HM_volume); % 使用SVM算法拟合血肿体积和时间的关系
        b = [mdl.Bias; mdl.Beta(:)]; % 获取SVM模型的系数存储到b中
    else % 如果该病人没有血肿体积数据，则使用样本数据匹配最接近的其他样本对应的参数
        AA=mapminmax(A',0,1)';
        d=pdist2(AA(end,:),AA(1:end-1,:));
        [~,o]=min(d);
        b=B(:,o);
    end
    B=[B,b.'];
    t1=[t(1):0.1:48]'; % 将时间网格化，精度为0.1
    HM_volume_48=[ones(length(t1),1),t1]*b; % 时间设为第一次诊断到发病第48小时内
    % 找到血肿相对扩张大于0.33或绝对扩张大于6000的时间点，合并到aa中
    a1=find(HM_volume_48./HM_volume(1)>1.33); 
    a2=find(HM_volume_48-HM_volume(1)>6000);
    aa=union(a1,a2);
    if length(aa)>0 % 发生了血肿扩张，将Y设为1，将血肿扩张时间存储到T中
        Y(i,1)=1;
        if aa(1)<=length(t1)
            T(i,1)=t1(aa(1));
        else
            T(i,1)=0;
        end
    else
        Y(i,1)=0;
        T(i,1)=0;
    end
end
result1=[Num,Y,T]; % 将Num、Y和T存储到result1中

%% b
In=mapminmax(A',0,1); % 将A和Y进行归一化处理
Out=Y';
% bp神经网络，划分训练集和测试集
Xtrain = In(:,1:100);
Ytrain = Out(:,1:100);
Xtest1 = In(:,101:130);
Ytest1= Out(:,101:130);
Xtest2 = In(:,131:160);
Ytest2= Out(:,131:160);
% 1. 创建一个前馈神经网络，其中输入层有size(Xtrain,1)个神经元，隐藏层有fix(size(Xtrain,1)/2)和fix(size(Xtrain,1)/4)个神经元，输出层有1个神经元。激活函数为双曲正切函数
net = newff(Xtrain,Ytrain,[fix(size(Xtrain,1)/2),fix(size(Xtrain,1)/4)],{'tansig','tansig'});
% 2. 设置训练参数
net.trainParam.epochs = 1000; % 最大迭代次数，到达最大迭代次数则终止
net.trainParam.goal = 1e-100; % 训练误差，达到目标误差则终止
net.trainParam.min_grad = 1e-100; % 性能函数的最小梯度
net.trainParam.lr = 1e-5; % 学习率
net.trainParam.max_fail=100; % 最大确认失败次数，终止条件之一
% 3. 训练网络
net = train(net,Xtrain,Ytrain);
% 4. 用训练集和测试集对神经网络进行仿真测试，存储仿真结果
t_sim = max(sim(net,Xtrain),0);
t_sim1 = max(sim(net,Xtest1),0);
t_sim2 = max(sim(net,Xtest2),0);
T_sim = round(t_sim);
T_sim1 = round(t_sim1);
T_sim2 = round(t_sim2);
% 绘制混淆矩阵
figure
plotconfusion(categorical(Ytrain),categorical(T_sim));
title('训练集')
figure
plotconfusion(categorical(Ytest1),categorical(T_sim1));
title('测试集1')
figure
plotconfusion(categorical(Ytest2),categorical(T_sim2));
title('测试集2')

result1=[["患者","是否发生血肿扩张","血肿扩张时间","血肿扩张预测概率"];[result1,[t_sim';t_sim1';t_sim2']]]; 