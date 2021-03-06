addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;

%% Caffe model setup. 
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = 'caffe/verydeep_deploy_L1.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
% model_file = '/research/action_videos/shared/caffe-dev/models/vgg_very_deep/VGG_ILSVRC_16_layers.caffemodel';
matcaffe_init(1, model_def_file, model_file);

%% Params that may need to be changed for each run.
IMAGE_DIM = 224;
bz = 10;
layer_name = 'fc7';
net_name = 'ucf101vgg16K1';

%% Finds top activations for each unit (i.e. CNN filter). 
% Video frames.
load(['data' filesep 'vframe_vis_sample.mat']);
F_DIM = 4096;
A = zeros(length(img_names), F_DIM);

for i = 1:bz:length(img_names)
    batch_images = zeros(IMAGE_DIM, IMAGE_DIM, 3, bz, 'single');
    for j = 1:bz
        id = i + j - 1;
        im = single(imread(img_names{id}));
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        if ndims(im) == 2
            im = cat(3, im, im, im);
        end
        im = transform_image(im);
        im(:, :, 1) = im(:, :, 1) - single(103.939);
        im(:, :, 2) = im(:, :, 2) - single(116.779);
        im(:, :, 3) = im(:, :, 3) - single(123.68);
        batch_images(:, :, :, j) = im;
    end
    cnn_input = {batch_images};
    s = caffe('forward', cnn_input);
    s = squeeze(s{1});
    A(i:i+bz-1, :) = s';
    if mod(i, 1000) == 1
        fprintf('Processed %d video frames\n', i-1);
    end
end
save(['/research/action_videos/video_data/deepnet_ucf101/visual_data/activation_vframe_' net_name '_' layer_name], 'A', '-v7.3');

% Web images.
load(['data' filesep 'webimg_vis_sample.mat']);
A = zeros(length(img_names), F_DIM);
for i = 1:bz:length(img_names)
    batch_images = zeros(IMAGE_DIM, IMAGE_DIM, 3, bz, 'single');
    for j = 1:bz
        id = i + j - 1;
        im = single(imread(img_names{id}));
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        if ndims(im) == 2
            im = cat(3, im, im, im);
        end
        im = transform_image(im);
        im(:, :, 1) = im(:, :, 1) - 103.939;
        im(:, :, 2) = im(:, :, 2) - 116.779;
        im(:, :, 3) = im(:, :, 3) - 123.68;
        batch_images(:, :, :, j) = im;
    end
    cnn_input = {batch_images};
    s = caffe('forward', cnn_input);
    s = squeeze(s{1});
    A(i:i+bz-1, :) = s';
    if mod(i, 1000) == 1
        fprintf('Processed %d web images\n', i-1);
    end
end
save(['/research/action_videos/video_data/deepnet_ucf101/visual_data/activation_webimg_' net_name '_' layer_name], 'A', '-v7.3');

% ImageNet images.
load('imgnet_imgs_select.mat');
A = zeros(length(imgnet_fnames), F_DIM);
for i = 1:bz:length(imgnet_fnames)
    batch_images = zeros(IMAGE_DIM, IMAGE_DIM, 3, bz, 'single');
    for j = 1:bz
        id = i + j - 1;
        im = single(imread(imgnet_fnames{id}));
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        if ndims(im) == 2
            im = cat(3, im, im, im);
        end
        im = transform_image(im);
        im(:, :, 1) = im(:, :, 1) - 103.939;
        im(:, :, 2) = im(:, :, 2) - 116.779;
        im(:, :, 3) = im(:, :, 3) - 123.68;
        batch_images(:, :, :, j) = im;
    end
    cnn_input = {batch_images};
    s = caffe('forward', cnn_input);
    s = squeeze(s{1});
    A(i:i+bz-1, :) = s';
    if mod(i, 1000) == 1
        fprintf('Processed %d imagenet images\n', i-1);
    end
end
save(['/research/action_videos/video_data/deepnet_ucf101/visual_data/activation_imgnet_' net_name '_' layer_name], 'A', '-v7.3');
