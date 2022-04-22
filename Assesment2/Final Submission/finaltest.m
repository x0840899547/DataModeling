clear all
close all
clc

%load iris data
versicolor = load("iris_versicolor.txt");
virginica = load("iris_virginica.txt");
setosa = load("iris_setosa.txt");
k=5;
% Put the three species into a data matrix.
% Just use two dimensions (1 and 2) for now.
X = [versicolor(:,1:2);virginica(:,1:2);setosa(:,1:2)];

% Add a third column that has the class labels.
% label 0 for versicolor, and 1 for virginica. two for setosa
X(:,3) = [zeros(50,1);ones(50,1);2*ones(50,1)];

% Designate a point in the space to classify.
% Make the third element '99' to indicate
% missing class label.
x = [7.0, 4.0, 99];
% Create a matrix with the new observation
% as the first row and the remaining ones as the
% observed data.
Xn = [x ; X];

% First, get the pairwise distances.
D = squareform(pdist(Xn(:,1:2)));
% Then, find the ones closest to the new point.
% Do this by looking in the first row or column.
[pd, ind ] = sort(D(:,1));

% Find the frequency of classes within the 3
% nearest neighbors.
tabulate(Xn(ind(2:6),3));

KNDist = pd(2:k+1,:);
%Mdl build in fuction to give train result of neareast neighbor
Mdl = fitcknn(X(:,1:2),X(:,3),'NumNeighbors',5,'Standardize',1)
label = predict(Mdl,X(:,1:2));


perform=sum(X(:,3)==label)/length(label)*100
figure(1)
hold on

gscatter(X(:,1),X(:,2),X(:,3));
plot(x(1),x(2),'o');
theta = linspace(0,2*pi);
xi = KNDist(k,1)*cos(theta) + Xn(1,1);
yi = KNDist(k,1)*sin(theta) + Xn(1,2);
plot(xi,yi,'-');

% Add legend, label, and title.
title('FisherIris Classification');
legend('versicolor','verginica','setosa','Sample to evaluate');
xlabel('sepal length');
ylabel('sepal width');
axis([4 8 2 5])
hold off

%Training Set for 70 train and 30 Test
train_set = [versicolor(1:35,1:2);virginica(1:35,1:2);setosa(1:35,1:2)];
twos = 2.*ones(35,1);
%set number for each species
train_set(:,3) = [zeros(35,1);ones(35,1);twos];
%set number for each testing for 15 percent

test_set = [versicolor(36:end,1:2);virginica(36:end,1:2);setosa(36:end,1:2)];

test_set(:,3) = [zeros(15,1);ones(15,1);2.*ones(15,1)];
scores = zeros(length(test_set),3);
for i = 1:length(test_set)
    point = [test_set(i,1),test_set(i,2), 99];
    train_set_n = [point ; train_set];
    D = squareform(pdist(train_set_n(:,1:2)));
    [pd, ind ] = sort(D(:,1));
    tbl = tabulate(train_set_n(ind(2:k+1),3));
    for j = 1:height(tbl)
        scores(i,(tbl(j,1))+1) = tbl(j,3);
    end
end
%plot ROC
figure(2)
hold on
[X0,Y0,T0,AUC0] = perfcurve(test_set(:,3),scores(:,1),0);
plot(X0,Y0,'--','MarkerEdgeColor','auto')
[X1,Y1,T1,AUC1] = perfcurve(test_set(:,3),scores(:,2),1);
plot(X1,Y1,':','MarkerEdgeColor','auto')
[X2,Y2,T2,AUC2] = perfcurve(test_set(:,3),scores(:,3),2);
plot(X2,Y2,'MarkerEdgeColor','auto')
hold off
%Add Label for all axis and title and a legend
xlabel('False Positive Rate'); 
ylabel('True Positive Rate');
title('ROC curve of K-Nearest Neighbor');
legend('Versicolor = 0', 'Virginica = 1', 'Setosa = 2')





