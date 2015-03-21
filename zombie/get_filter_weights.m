addpath('..');
addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;

%% Caffe model setup. 
use_gpu = 0;
% model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
% model_def_file = '../caffe/verydeep_deploy.prototxt';
% model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
% matcaffe_init(use_gpu, model_def_file, model_file);
% W = caffe('get_weights');
% W = W(13).weights;
% save('conv5-3-tuned-weights.mat', 'W');

caffe('reset');
model_path = ['/research/action_videos/shared/caffe-dev/models/vgg_very_deep'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'VGG_ILSVRC_16_layers.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);
W = caffe('get_weights');
W = W(13).weights;
save('conv5-3-org-weights.mat', 'W');


