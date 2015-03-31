W1 = load([layer_name '-tuned-weights.mat']);
W2 = load([layer_name '-org-weights.mat']);
[w nchs] = size(W1.W{1}); 

for i = 1:nchs
    x = W1.W{1}(:, i);
    x = [x(:); W1.W{2}(i)];
    
    y = W2.W{1}(:, i);
    y = [y(:); W2.W{2}(i)];

    D(i) =sqrt(sum((x - y) .* (x - y)));
end
save([layer_name '-filter-change.mat'], 'D');
plot(1:nchs, sort(D), 'Color', 'r', 'LineWidth', 3);
xlabel('filter number');
ylabel('filter change');
title(['filter changes (sorted) of ' layer_name]); 

