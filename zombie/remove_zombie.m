addpath('..');
addpath('/research/action_videos/caffe_jm/caffe/matlab/caffe');

model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
model_file_modified = [model_path filesep ...
    'ucf101augVGG16K1All_iter_80000_modified.caffemodel']
matcaffe_init(0, model_def_file, model_file);
W = caffe('get_weights');
conv53_W = W(13).weights;

load('zombie_filter_conv5_3.mat');
for i = 1:length(filter_idx)
    W(13).weights{2}(i) = 0;
    W(13).weights{1}(:, :, :, i) = randn(3, 3, 512) * 0.01;
end
caffe('set_weights', W);
caffe('save', model_file_modified);
