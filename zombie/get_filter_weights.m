addpath('..');
addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab'));
init_ucf101;
visual_root = pathstring('X:\video_data\deepnet_ucf101\visual_data');

%% Caffe model setup. 
use_gpu = 0;
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels/ucf'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'ucf101augVGG16Split1_iter_80000.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);
W1 = caffe('get_weights');
W = W1(4).weights;
save([visual_root filesep 'conv2-2-tuned-weights.mat'], 'W');
W = W1(7).weights;
save([visual_root filesep 'conv3-3-tuned-weights.mat'], 'W');
W = W1(10).weights;
save([visual_root 'conv4-3-tuned-weights.mat'], 'W');

caffe('reset');
model_path = ['/research/action_videos/shared/caffe-dev/models/vgg_very_deep'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'VGG_ILSVRC_16_layers.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);
W1 = caffe('get_weights');
W = W1(4).weights;
save([visual_root filesep 'conv2-2-org-weights.mat'], 'W');
W = W1(7).weights;
save([visual_root filesep 'conv3-3-org-weights.mat'], 'W');
W = W1(10).weights;
save([visual_root filesep 'conv4-3-org-weights.mat'], 'W');


