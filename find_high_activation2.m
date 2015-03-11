addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;

%% Caffe model setup. 
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = 'caffe/verydeep_deploy_L1.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
matcaffe_init(1, model_def_file, model_file);

%% Params that may need to be changed for each run.
IMAGE_DIM = 224;
bz = 10;
nchs = 512;
layer_name = 'conv5_3';
net_name = 'ucf101vgg16K1';

%% Finds top activations for each unit (i.e. CNN filter). 
% Video frames.
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
        im = transform_image(im);
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
save(['visual_data/activation_vframe_' net_name '_' layer_name], 'A');

% Web images.
load(['data' filesep 'webimg_vis_sample.mat']);
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
        im = transform_image(im);
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
save(['visual_data/activation_webimg_' net_name '_' layer_name], 'A');


% ImageNet images.
load('imgnet_imgs_select.mat');
A = zeros(length(imgnet_fnames), nchs * 2);
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
    s = s{1};
    for j = 1:bz
        id = i + j - 1;
        s1 = reshape(s(:,:,:,j), [], nchs);
        [max_val, ix] = max(s1);
        A(id, :) = [max_val ix];
    end
end
save(['visual_data/activation_imgnet_' net_name '_' layer_name], 'A');
