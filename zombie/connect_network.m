addpath('..');
addpath('/research/action_videos/caffe_jm/caffe/matlab/caffe');

model_file = '/research/action_videos/caffe_v1.0/caffe/models/ucf101_very_deep/ucf101augAllM5253_fc6out_iter_80000.caffemodel';
model_def_file = '/research/action_videos/caffe_v1.0/caffe/models/ucf101_very_deep/deploy_layerwise.prototxt';
matcaffe_init(0, model_def_file, model_file);
W1 = caffe('get_weights');
caffe('reset');

model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = '../caffe/verydeep_deploy.prototxt';
model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
matcaffe_init(0, model_def_file, model_file);
W2 = caffe('get_weights');

W2(12) = W1(12);
W2(13) = W1(13);

model_file_modified = [model_path filesep ...
    'ucf101augVGG16K1All_iter_80000_modified5352LW.caffemodel']
caffe('set_weights', W2);
caffe('save', model_file_modified);
