load(['data' filesep 'vframe_vis_sample_k10.mat']);
load(['data' filesep 'activation_K10_conv5_3.mat']);
nchs = 512;
rf = 211;
stride = 16;
map_dim = 14;

K = 100;
img_dim = 224;
hrf = floor(rf / 2);

rfim_path = ['visualize' filesep 'ucf101_augK10vgg16All' filesep 'conv5-3'];

if ~exist(rfim_path, 'file')
    mkdir(rfim_path);
end

S = A(:, 1:nchs);
[S_sorted, IX] = sort(S, 'descend');
mean_rfim = cell(nchs, 1);
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
        rfim = im(y : y + rf - 1, x : x + rf - 1, :);
        mean_rfim{i} = mean_rfim{i} + rfim / K;
%         subplot(1, 2, 1);
%         imshow(im1);
%         subplot(1, 2, 2);
%         imshow(uint8(rfim));
%         pause(0.1);
    end
    imwrite(uint8(mean_rfim{i}), [rfim_path filesep num2str(i) '.png'], 'png'); 
end

