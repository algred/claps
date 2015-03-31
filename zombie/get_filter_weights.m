addpath('..');
addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;

%% Caffe model setup. 
use_gpu = 0;
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);
W1 = caffe('get_weights');
W = W1(14).weights;
save('fc6-tuned-weights.mat', 'W');
W = W1(15).weights;
save('fc7-tuned-weights.mat', 'W');

caffe('reset');
model_path = ['/research/action_videos/shared/caffe-dev/models/vgg_very_deep'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'VGG_ILSVRC_16_layers.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);
W1 = caffe('get_weights');
W = W1(14).weights;
save('fc6-org-weights.mat', 'W');
W = W1(15).weights;
save('fc7-org-weights.mat', 'W');


