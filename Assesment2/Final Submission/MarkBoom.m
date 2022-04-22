versicolor = load("iris_versicolor.txt");
virginica = load("iris_virginica.txt");
setosa = load("iris_setosa.txt");
num_train = 0.7.*length(versicolor);
train_set = [versicolor(1:num_train,1:2);virginica(1:num_train,1:2);setosa(1:num_train,1:2)];
twos = 2.*ones(num_train,1);
train_set(:,3) = [zeros(num_train,1);ones(num_train,1);twos];
num_test = 0.3.*length(versicolor);
test_set = [versicolor(num_train+1:end,1:2);virginica(num_train+1:end,1:2);setosa(num_train+1:end,1:2)];
twos = 2.*ones(num_test,1);
test_set(:,3) = [zeros(num_test,1);ones(num_test,1);twos];
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
hold on
[X0,Y0,T0,AUC0] = perfcurve(test_set(:,3),scores(:,1),0);
plot(X0,Y0,'b')
[X1,Y1,T1,AUC1] = perfcurve(test_set(:,3),scores(:,2),1);
plot(X1,Y1,'r')
[X2,Y2,T2,AUC2] = perfcurve(test_set(:,3),scores(:,3),2);
plot(X2,Y2,'y')
hold off
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve of KNN');
legend('0 = versicolor', '1 = virginica', '2 = setosa')