clc

% 读取数据
[~,~,data1]=xlsread('表1-患者列表及临床信息.xlsx','A1:W161');
data1=string(data1);
data1(find(ismissing(data1)==1))="";  % 将缺失值替换为空字符串
% 将血压数据拆分为最大值和最小值，并将其添加到data1中
data1=[data1(:,1:15),[["血压最大值","血压最小值"];split(data1(2:end,16),"/")],data1(:,17:end)]; 
% 用1和2代替性别数据
data1(find(data1=="男"))="1";
data1(find(data1=="女"))="2";

[~,~,data2]=xlsread('表2-患者影像信息血肿及水肿的体积及位置.xlsx','A1:GZ161');
data2=string(data2);
data2(find(ismissing(data2)==1))="";

[~,~,data3]=xlsread('表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx','B1:AG577');
data3=string(data3);
data3(find(ismissing(data3)==1))="";

[~,~,data4]=xlsread('附表1-检索表格-流水号vs时间.xlsx','C2:AB161');
data4=string(data4);
data4(find(ismissing(data4)==1))="";

% 整理数据
Table=["患者","流水号","时间/小时",data1(1,[5:14,16:24]),data2(1,3:24),data3(1,2:32)]; % 记录指标数据
for i=1:size(data1,1)-1 % 逐行遍历data1
    delta_t0=double(data1(i+1,15)); % 从发病到第一次诊断时的时间长度
    a=find(data4==data1(i+1,4)); % 查找data4中与当前行数据中的流水号匹配的行
    a1=mod(a,160); % 计算匹配行在data4中的列数
    if a1==0 % 如果列数为0，则说明匹配行在data4中的最后一列，将列数设置为160
        a1=160;
    end
    a2=ceil(a/160); % 计算匹配行在data4中的行数
    t1=data4(a1,a2-1); % 获取第一次诊断日期
    table1=data1(i+1,[5:14,16:24]); % 获取当前行的临床信息数据
    table2=data2(i+1,2:208); % 获取当前行的影像信息数据

    % 识别检查了多少次
    a=find(table2==""); % 查找table2中为空字符串的位置
    if length(a)==0 % 如果没有空字符串，则说明检查了23次
        a=length(table2);
    else % 否则，检查次数为第一个空字符串的位置
        a=a(1);
    end
    n=(a-1)/23; % 计算检查次数

    % 统计每次检查的数据，如果有缺失则不记录
    for j=1:n % 遍历每次检查
        a=find(data4==table2((j-1)*23+1)); % 查找data4中与当前检查匹配的行
        a1=mod(a,160); % 计算匹配行在data4中的列数
        if a1==0 % 如果列数为0，则说明匹配行在data4中的最后一列，将列数设置为160
            a1=160;
        end
        a2=ceil(a/160); % 计算匹配行在data4中的行数
        t2=data4(a1,a2-1); % 获取当前诊断日期
        time=max((double(t2)-double(t1))*24+delta_t0,0); % 以刚发病时刻为0，计算时间轴上第j次随诊距离发病的时间长度
        if length(find(data3(:,1)==table2((j-1)*23+1))>0) % 如果影像信息中有当前检查的数据，将数据添加到x中
            x=[data1(i+1,1),table2((j-1)*23+1),time,table1,table2((j-1)*23+2:j*23),data3(find(data3(:,1)==table2((j-1)*23+1)),2:end)];
        else % 利用已有的数据来填补缺失值
            x1=[double(table2((j-1)*23+2:j*23));double(Table(2:end,23:44))]; % 将当前检查的数据和之前的数据合并
            x1=mapminmax(x1',0,1)'; % 数据标准化
            d=pdist2(x1(1,:),x1(2:end,:)); % 计算当前检查数据与之前数据的距离
            [~,o]=sort(d); % 距离排序
            o=o(1:min(3,length(o))); % 取前三个距离最近的数据
            x=[data1(i+1,1),table2((j-1)*23+1),time,table1,table2((j-1)*23+2:j*23),mean(double(Table((o+1),45:75)),1)]; % 将数据添加到x中
        end
        Table=[Table;x]; % 将x添加到Table中
    end
end

save data Table