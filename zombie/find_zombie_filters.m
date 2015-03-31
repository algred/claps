addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));

layer_name = 'fc6';
net_name = 'ucf101vgg16K1';
data_path = '/research/action_videos/video_data/deepnet_ucf101/'

A1 = load([data_path filesep 'visual_data' filesep ...
    'activation_vframe_' net_name '_' layer_name]);
A2 = load([data_path filesep 'visual_data' filesep ...
    'activation_webimg_' net_name '_' layer_name]);
A3 = load([data_path filesep 'visual_data' filesep ...
    'activation_imgnet_' net_name '_' layer_name]);
S = [A1.A; A2.A; A3.A];
source_id = [ones(size(A1.A, 1), 1); ones(size(A2.A, 1), 1) * 2; ones(size(A3.A, 1), 1) * 3];
n = length(source_id);

N = 5000; 
% parfor i = 1:size(S, 2)
    [s, ix] = sort(S(:, i), 'descend');
    source_id1 = source_id(ix);
    imgnet_idx = ix(source_id1 == 3);
    select_idx = imgnet_idx(1:N);
    
    y = ones(n, 1) * -1;
    y(select_idx) = 1;

    [recall, precision, info] = vl_pr(y, S(:, i));
    ap(i) = info.ap_interp_11;
end

load([layer_name '-filter-change']);
[D_sorted, ix] = sort(D);

bz = 64;
ap_sorted = ap(ix);
for i = 1:floor(length(D_sorted) / bz)
    d(i) = mean(D_sorted((i -1) * bz + 1 : i * bz));
    a(i) = mean(ap_sorted((i -1) * bz + 1 : i * bz));
end


hold on;
plot(d, a, 'Color', 'r', 'LineWidth', 3);
% plot(D_sorted, ap1(ix), 'Color', 'r');
% plot(D_sorted, ap2(ix), 'Color', 'b');
% plot(D_sorted, ap(ix), 'Color', 'g');




