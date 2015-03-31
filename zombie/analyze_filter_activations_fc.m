layer_name = 'fc6';
net_name = 'ucf101vgg16K1';
data_source = 'vframe'

vframe = load(['../data' filesep 'vframe_vis_sample.mat']);
A1 = load(['/research/action_videos/video_data/deepnet_ucf101/visual_data'...
    filesep 'activation_' data_source '_' net_name '_' layer_name]);
S = A1.A;
[S_sorted, IX] = sort(S, 'descend');
intv = 200;
N = floor(size(S_sorted, 1) / intv);
for i = 1:N
    S_avg(i, :) = mean(S_sorted((i -1) * intv + 1 : i * intv, :)); 
end
S_avg = S_avg';
hold on;
colors = {'r', 'g', 'b'};
load('fc6-filter-change.mat');

K = 4;
thre = [0 0.075 0.125 0.25];
hold on;
for i = 2:K
    val = mean(S_avg(D < thre(i) & D >= thre(i-1), :), 1);
    plot(1:100, val(1:100), 'Color', colors{i-1}, 'LineWidth', 3);
end
xlabel('#sample x 200');
ylabel('activation');
legend('filter-change < 0.075', 'filter-change < 0.125', ...
    'filter-change > 0.125'); 
title(['Activation patterns of clusters of ' layer_name ' on ' data_source]);
save(['fc6-activations-' data_source], 'S_avg');


