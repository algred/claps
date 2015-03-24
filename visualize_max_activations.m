load(['data' filesep 'vframe_vis_sample.mat']);
% layer_name = 'conv5_3';
rf = 211;
prefix = 'vframe';
net_name = 'ucf101vgg16K1';
nchs = 512;
stride = 16;
map_dim = 14;

load(['visual_data' filesep 'activation_' prefix '_' net_name '_' layer_name]);


img_dim = 224;
hrf = floor(rf / 2);

rfim_path = ['visualize' filesep net_name filesep prefix filesep layer_name];

if ~exist(rfim_path, 'file')
    mkdir(rfim_path);
end

S = A(:, 1:nchs);
[S_sorted, IX] = sort(S, 'descend');
mean_rfim = cell(nchs, 1);
K = 200;
parfor i = 1:nchs
    mean_rfim{i} = zeros(rf, rf, 3);
    idx = IX(1:K, i);
    for j = 1:K
        id = idx(j);
        im = zeros(img_dim + hrf * 2, img_dim + hrf * 2, 3);
        im1 = imread(pathstring(img_names{id}));
        im1 = imresize(im1, [img_dim, img_dim], 'bilinear');
        im(hrf + 1 : hrf + img_dim, hrf + 1 :  hrf + img_dim, :) = im1;
        uid = A(id, nchs + i);
        [ur, uc] = ind2sub([map_dim, map_dim], uid);
        x = (uc - 1) * stride + 1;
        y = (ur - 1) * stride + 1;
        rfim = im(x : x + rf - 1, y : y + rf - 1, :);
        mean_rfim{i} = mean_rfim{i} + rfim / K;
    end
    imwrite(uint8(mean_rfim{i}), [rfim_path filesep num2str(i) '.png'], 'png'); 
end

