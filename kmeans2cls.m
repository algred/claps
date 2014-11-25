function [t, x1, x2, minD] = kmeans2cls(X)
X = X(:);
y = unique(round(X));
d = zeros(size(y));
for i = 1:length(y)
    yy = y(i);
    dx = [X(X <= yy) - yy; X(X > yy) - yy];
    d(i) = sum(dx .* dx);
end
[minD, ix] = min(d);
t = y(ix);
x1 = sum((X(X <= t) - t) .* (X(X <= t) - t));
x2 = sum((X(X > t) - t) .* (X(X > t) - t));
end