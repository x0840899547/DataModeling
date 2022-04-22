% Define function a is number of k. b,c is feature
function [X,Y]=TestModel(a,b,c)
% load data, set k value
setosa = load("iris_setosa.txt");
versicolor = load("iris_versicolor.txt");
virginica = load("iris_virginica.txt");
k=a;

% Set train amount, select feature to classify
train=70*50/100;
feature1=b;
feature2=c;

% Build train set
setosatrain=[setosa(1:train,feature1) setosa(1:train,feature2)];
versitrain=[versicolor(1:train,feature1) versicolor(1:train,feature2)];
virgitrain=[virginica(1:train,feature1) virginica(1:train,feature2)];

% Build test set
setosatest=[setosa(train+1:50,feature1) setosa(train+1:50,feature2)];
versitest=[versicolor(train+1:50,feature1) versicolor(train+1:50,feature2)];
virgitest=[virginica(train+1:50,feature1) virginica(train+1:50,feature2)];

% merge all train and test set.
trainset = [setosatrain;versitrain;virgitrain];
testset = [setosatest;versitest;virgitest];

% Add a third column that has the class labels.
% label -1 for setosa, 0 for versicolor, and 1 for virginica.
temp = zeros(train,1)-ones(train,1);
temp2 = zeros(50-train,1)-ones(50-train,1);
temp3 = zeros((50-train)*3,1)+99;
trainset(:,3) = [temp;zeros(train,1);ones(train,1)];
outputset = testset;
outputset(:,3) = temp3;

% Create matrix to receive the frequency of each class from KNN
% These matrix will be feed to roc function
ROCoutput=zeros((50-train)*3,3);
ROCtarget=zeros((50-train)*3,3);
ROCtarget(1:15,1)=1;
ROCtarget(16:30,2)=1;
ROCtarget(31:45,3)=1;

% Test the moedl with each element in testset
for i=1:45
    Xn = [outputset(i,:) ; trainset];
    D = squareform(pdist(Xn(:,1:2)));
    [pd, ind ] = sort(D(:,1));
    result=tabulate(Xn(ind(2:k+1),3));
    if size(result,1)==1
        ROCoutput(i,result(1)+2)=result(3)/100;
    else
        for j=1:size(result,1)
            ROCoutput(i,result(j,1)+2)=result(j,3)/100;
        end
    end
    
    outputset(i,3) = mode(Xn(ind(2:k+1),3)); 
end

% use builted-in roc function to compute True positive rate and False
% positive rate (It is add-on in Deep Learning Toolbox)
% Hope you let me use it krub ajarn :)
[X,Y,~]=roc(ROCtarget.',ROCoutput.');