W1 = load('conv5-2-tuned-weights.mat', 'W');
W2 = load('conv5-2-org-weights.mat', 'W');
[w h z nchs] = size(W1.W{1}); 

for i = 1:nchs
    x = W1.W{1}(:,:,:,i);
    x = [x(:); W1.W{2}(i)];
    
    y = W2.W{1}(:,:,:,i);
    y = [y(:); W2.W{2}(i)];

    D(i) =sqrt(sum((x - y) .* (x - y)));
end
save('conv5-2-filter-change.mat', 'D');
plot(1:nchs, sort(D), 'Color', 'r', 'LineWidth', 3);
xlabel('filter number');
ylabel('filter change');
title('filter changes (sorted) of conv5-2'); 

