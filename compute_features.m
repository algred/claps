addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab/caffe'));
frame_path = '/research/action_data/ucf101-frames/';
init_ucf101;

% model_path = '/research/action_videos/caffe_v1.0/caffe/models/ucf101_vgg16_fuse4';
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = ['caffe/verydeep_deploy_L1.prototxt']
model_file = [model_path filesep 'ucf101augVGG16Split2_iter_80000.caffemodel']
matcaffe_init(1, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101/cnnfeat/split2/fc6';
IMG_DIM = 224;
FEAT_DIM = 4096;
N = 20;
fprintf('Computed features of video:\n');
for vid = 11771:length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4);
    intv = max(1, floor(nfms / N));
    idx = 1:intv:nfms;
    S = zeros(FEAT_DIM, length(idx) * 10);
    for i = 1 : length(idx)
        fid = idx(i);
        imgs = prepare_image_vgg(frames.imgs(:,:,:,fid), IMG_DIM);
        input_data = {imgs};
        s = caffe('forward', input_data);
        S(:, (i-1) * 10 + 1 : i * 10) = squeeze(s{1});
    end
    S = S';
    fprintf('%d\n', vid);
    save([out_path filesep num2str(vid) '_fc6.mat'], 'S');
end

