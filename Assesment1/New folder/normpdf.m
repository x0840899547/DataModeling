function plot_fig_Nor(n,p,N)

% Call function for generating random samples
X = randn(n,N);

%Generate a figure
figure;
%Plot a relative frequency histogram for the randomly generated samples
[N1,X1]=hist(X(:));

bar(X1,N1/length(X(:)))
hold on;

x1=-10:1:10;

mu=0;
sigma=p;
 y1 = exp(-0.5 * ((x1 - mu)./sigma).^2) ./ (sqrt(2*pi) .* sigma);
%Plot the theoretical values
plot(x1,y1)
xlabel('Number of successes') % label x-axis
ylabel('Relative Frequency') % label y-axis
title(['Standard normal distribution Sampling n = ' num2str(n)]) % Figure title
legend('Random Generation','Theoretical Value') %label the legend
end