clear all
close all
clc


data_iris

data1=[];
for ii=1:size(A,1)
    data1=[data1;A{ii,1:4}]
    output{ii,1}=A{ii,5};
end


input0=[data1(:,3:4)];
% output=[1*ones(size(A12,1),1);2*ones(size(A22,1),1);3*ones(size(A32,1),1)];

input=input0;
output1=[ones(50,1);2*ones(50,1);3*ones(50,1)];
data=[input,output1];

X=input;

% plot(output,input,'*m')

Mdl = fitcknn(input,output1,'NumNeighbors',5,'Standardize',1)
% X=[5.5000    2.4000    3.7000    1.0000]
label = predict(Mdl,X);
[label_1,score,cost] = predict(Mdl,input)


k = 5;
metric = 'euclidean';

mdl = kNNeighbors(k,metric);
mdl = mdl.fit(input,output1);
% 
% Xnew = [min(X);mean(X);max(X)];

Xnew = [mean(X(1:50,:));mean(X(61:100,:));mean(X(101:150,:))];
ax=1;
ay=2;

[distances,indices] = mdl.find(Xnew);
figure(2)
hold on
plot(X(1:50,ax),X(1:50,ay),'o') % only setosas
plot(X(51:100,ax),X(51:100,ay),'o') % only versicolors
plot(X(101:150,ax),X(101:150,ay),'o') % only virginics
plot(Xnew(:,ax),Xnew(:,ay),'xk')

legend({'setosa','versicolor','virginica','new instances'},'Location','northwest')
distances(5,1)=1;
distances(5,2)=1;
distances(5,3)=1;
theta = linspace(0,2*pi);
in1=[];
out1=[];
for i = 1:3
    x = distances(k,i)*cos(theta) + Xnew(i,ax);
    y = distances(k,i)*sin(theta) + Xnew(i,ay);
    plot(x,y,'--k')
    in1=[in1;x' y'];
    out1=[out1;i*ones(size(x,2),1)];
end

axis equal
hold off

figure(3)
                                 %
% roc_curve(output1, label_1);  

%    [tpr,fpr,th] = roc(output1, label_1);
   
%    
% in1=[x' y'];
% label = predict(Mdl,in1);
%  plotroc(out1,label)

plotroc(output1, label_1)