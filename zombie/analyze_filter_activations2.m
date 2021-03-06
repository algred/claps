layer_name = 'conv4_3';
layer_name2 = 'conv4-3';
net_name = 'ucf101vgg16K1';
data_root = pathstring('V:\deepnet\data');
visual_root = pathstring('X:\video_data\deepnet_ucf101\visual_data');
% vframe = load([data_root filesep 'vframe_vis_sample.mat']);
A1 = load([visual_root filesep 'activation_vframe_' net_name '_' layer_name]);
% A1 = load([visual_root filesep 'activation_webimg_' net_name '_' layer_name]);
nchs = 512;
S = A1.A(:, 1:nchs);
[S_sorted, IX] = sort(S, 'descend');
intv = 200;
N = floor(size(S_sorted, 1) / intv);
for i = 1:N
    S_avg(i, :) = mean(S_sorted((i -1) * intv + 1 : i * intv, :)); 
end
S_avg = S_avg';
hold on;
colors = {'r', 'g', 'b'};
load([visual_root filesep layer_name2 '-filter-change.mat']);

K = 4;
thre = [0 0.075 0.125 0.25];
hold on;
for i = 2:K
    val = mean(S_avg(D < thre(i) & D >= thre(i-1), :), 1);
    plot(1:100, val(1:100), 'Color', colors{i-1}, 'LineWidth', 3);
end
set(gca,'YGrid','on');
xlabel('#sample (in uints of 200)');
ylabel('activation value');
legend('filter-change < 0.075', 'filter-change < 0.125', ...
    'filter-change > 0.125'); 
title('Activation patterns of clusters of CONV3-3');
% save([visual_root filesep layer_name2 '-activations.mat'], 'S_avg');

