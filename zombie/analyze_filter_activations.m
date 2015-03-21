layer_name = 'conv5_3';
net_name = 'ucf101vgg16K1';

vframe = load(['../data' filesep 'vframe_vis_sample.mat']);
A1 = load(['../visual_data' filesep 'activation_vframe_' net_name '_' layer_name]);
nchs = 512;
S = A1.A(:, 1:nchs);
[S_sorted, IX] = sort(S, 'descend');
intv = 200;
N = floor(size(S_sorted, 1) / intv);
for i = 1:N
    S_avg(i, :) = mean(S_sorted((i -1) * intv + 1 : i * intv, :)); 
end
S_avg = S_avg';
K = 4;
idx = kmeans(S_avg, K);
hold on;
colors = {'r', 'g', 'b', 'c'};
for i = 1:K
    val = mean(S_avg(idx == i, :));
    plot(1:100, val(1:100), 'Color', colors{i}, 'LineWidth', 3);
end
xlabel('#sample (in uints of 200)');
ylabel('activation value');
legend('cluster-1', 'cluster-2', 'cluster-3', 'cluster-4'); 
title('Activation patterns of clusters of CONV5-3');
save('conv5-3-clusters.mat', 'idx');



