function auc=plot_roc(predict,ground_truth)

%根据该数目可以计算出沿x轴或者y轴的步长
x_step = 1.0/length(predict);
y_step = 1.0/length(predict);
%首先对ground_truth中的分类器输出值按照从小到大排列
[ground_truth,index] = sort(ground_truth);
predict = predict(:,index);

for ii=1:size(predict,1)
    %对predict中的每个样本分别判断他们是FP或者是TP
    %遍历ground_truth的元素，
    %若ground_truth[i]=1,则TP增加了1，往y轴方向上升y_step
    %若ground_truth[i]=0,则FP增加了1，往x轴方向上升x_step
    %初始点为(0.0,0.0)
    x = 0.0;
    y = 0.0;
    X=[0];
    Y=[0];
    for i=1:length(ground_truth)
        if ground_truth(i) == predict(ii,i)
            y = y + y_step;
        else
            x = x + x_step;
        end
        X(i+1)=x;
        Y(i+1)=y;
    end
    X(end+1)=x;
    Y(end+1)=1;
    X(end+1)=1;
    Y(end+1)=1;
end
%画出图像     
plot(X,Y,'-','LineWidth',1);
%trapz，matlab自带函数，计算小矩形的面积,返回auc
auc(ii) = trapz(X,Y); 

end