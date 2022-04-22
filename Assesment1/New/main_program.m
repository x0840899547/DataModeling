%-------Main Program--------
% Plot 3 separate figures for Standard normal distribution of randomly generated samples and theoretical values.
clear all %clear data all in memory
close all %close all figure 
clc   % clear command window 
% Plot using the function normpdf 
% p=standard deviation
%n=5
normpdf(5,1,1000);
%n=25
normpdf(25,1,1000);
%n=50
normpdf(50,1,1000);



