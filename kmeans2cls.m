function [t c1 c2 minD] = kmeans2cls(X)
X = X(:)*10;
y = unique(round(X));
d = zeros(size(y));
for i = 1:length(y)
    yy = y(i);
    dx = [X(X <= yy) - mean(X(X <= yy)); X(X > yy) - mean(X(X > yy))] / 10;
    d(i) = sum(dx .* dx);
end
[minD, ix] = min(d);
t = y(ix) / 10;
c1 = mean(X(X <= t) / 10);
c2 = mean(X(X > t) / 10);
end
