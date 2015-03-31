addpath('..');
addpath('/research/action_videos/caffe_jm/caffe/matlab/caffe');

model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
model_file_modified = [model_path filesep ...
    'ucf101augVGG16K1All_iter_80000_modified53ctrl.caffemodel']
matcaffe_init(0, model_def_file, model_file);
W = caffe('get_weights');

load('zombie_filter_conv5_3.mat');
filter_idx = randsample(512, length(filter_idx));
for i = 1:length(filter_idx)
    id = filter_idx(i)
    W(13).weights{2}(id) = 0;
    W(13).weights{1}(:, :, :, id) = randn(3, 3, 512) .* 0.01;
end

% load('zombie_filter_conv5_3.mat');
% W1 = W(13).weights{1};
% [w, h, z, nchs] = size(W1);
% W1 = reshape(W1, [], nchs);
% sigma = std(W1');
% mu = mean(W1');
% sigma = reshape(sigma, w, h, z);
% mu = reshape(mu, w, h, z);
% sigma2 = std(W(13).weights{2});
% mu2 = mean(W(13).weights{2});
% for i = 1:length(filter_idx)
%     id = filter_idx(i)
%     W(13).weights{2}(id) = randn(1,1)*sigma2 + mu2;
%     W(13).weights{1}(:, :, :, id) = randn(3, 3, 512) .* sigma + mu;
% end

% load('zombie_filter_fc6.mat');
% filter_idx = randsample(4096, length(filter_idx));
% W1 = W(14).weights{1};
% [w, nchs] = size(W1);
% sigma = std(W1');
% mu = mean(W1');
% sigma = sigma(:);
% mu = mu(:);
% sigma2 = std(W(14).weights{2});
% mu2 = mean(W(14).weights{2});
% for i = 1:length(filter_idx)
%     id = filter_idx(i)
%     W(14).weights{2}(id) = randn(1,1)*sigma2+mu2;
%     W(14).weights{1}(:, id) = randn(w, 1) .* sigma + mu;
% end

% % load('zombie_filter_conv5_3_more.mat');
% for i = 1:length(filter_idx)
%     id = filter_idx(i)
%     W(13).weights{2}(id) = 0;
%     W(13).weights{1}(:, :, :, id) = randn(3, 3, 512) * 0.01;
% end

% load('zombie_filter_conv5_2.mat');
% for i = 1:length(filter_idx)
%     id = filter_idx(i)
%     W(12).weights{2}(id) = 0;
%     W(12).weights{1}(:, :, :, id) = randn(3, 3, 512) * 0.01;
% end

caffe('set_weights', W);
caffe('save', model_file_modified);
