addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;

%% Compiles or loads an image sample set.
% if ~exist(['data' filesep 'vframe_vis_sample_k10.mat']);
%     fid = fopen(['data' filesep 'ucf101_K10_train.txt'], 'r');
%     X = textscan(fid, '%s %d\n', -1);
%     fnames = X{1};
%     cidx = X{2};
%     fclose(fid);
%     N = 500;
%     ncls = 101;
%     sample_idx = zeros(N * ncls, 1);
%     for i = 1:ncls
%         cidx1 = find(cidx == (i-1));
%         ix = randsample(length(cidx1), N);
%         cidx1 = cidx1(ix);
%         sample_idx((i-1) * N + 1 : i * N) = cidx1(:);
%     end
%     img_names = fnames(sample_idx);
%     img_labels = cidx(sample_idx);
%     save(['data' filesep 'vframe_vis_sample_k10.mat'], 'img_names', 'img_labels');
% else
%     load(['data' filesep 'vframe_vis_sample_k10.mat']);
% end

%% Caffe model setup. 
use_gpu = 1;
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = 'caffe/verydeep_deploy_L1.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
% model_file = [model_path filesep 'ucf101augVGG16K10All_iter_80000.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);

%% Finds top activations for each unit (i.e. CNN filter). 
IMAGE_DIM = 224;
bz = 10;
nchs = 512;
load(['data' filesep 'vframe_vis_sample.mat']);
A = zeros(length(img_names), nchs * 2);
for i = 1:bz:length(img_names)
    batch_images = zeros(IMAGE_DIM, IMAGE_DIM, 3, bz, 'single');
    for j = 1:bz
        id = i + j - 1;
        im = single(imread(img_names{id}));
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        if ndims(im) == 2
            im = cat(3, im, im, im);
        end
        im = im(:,:,[3 2 1]);
        im(:, :, 1) = im(:, :, 1) - 103.939;
        im(:, :, 2) = im(:, :, 2) - 116.779;
        im(:, :, 3) = im(:, :, 3) - 123.68;
        batch_images(:, :, :, j) = im;
    end
    cnn_input = {batch_images};
    s = caffe('forward', cnn_input);
    s = s{1};
    for j = 1:bz
        id = i + j - 1;
        s1 = reshape(s(:,:,:,j), [], nchs);
        [max_val, ix] = max(s1);
        A(id, :) = [max_val ix];
    end
end
save('data/activation_ucf101vgg16K1_conv5_3.mat', 'A');
