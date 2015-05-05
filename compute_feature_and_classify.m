addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab/caffe'));
init_ucf101;
score_path = '/research/action_videos/video_data/deepnet_ucf101';

% Reads the weights of fc7, fc8.
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels/ucf'];
model_def_file = ['caffe/verydeep_deploy.prototxt']
model_file = [model_path filesep 'ucf101augVGG16Split'...
    num2str(split) '_iter_100000.caffemodel']
matcaffe_init(0, model_def_file, model_file);
W = caffe('get_weights');
W_fc7 = double(W(15).weights{1});
B_fc7 = double(W(15).weights{2}); 
B_fc7 = repmat(B_fc7(:)', 10, 1);

W_fc8 = double(W(16).weights{1});
B_fc8 = double(W(16).weights{2}); 
B_fc8 = repmat(B_fc8(:)', 10, 1);

caffe('reset');

% Computes output of fc6 and classifies the video.
model_def_file = ['caffe/verydeep_deploy_L1.prototxt']
caffe('set_device', device_id);
matcaffe_init(1, model_def_file, model_file);

% out_path = ['/research/action_videos/video_data/deepnet_ucf101/cnnfeat/split' ...
%    num2str(split) '/fc6'];

out_path = pathstring(['W:\ucf101_cnnfeat\split' ...
    num2str(split) filesep 'fc6'])

IMG_DIM = 224;
FEAT_DIM = 4096;
N = 20;
scores = cell(length(video_list), 1);
% split 2: 5270 temp_split2
% split 3: 5219 temp_split3
for vid = 1:length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4);
    intv = max(1, floor(nfms / N));
    idx = 1:intv:nfms;
    S = zeros(FEAT_DIM, length(idx) * 10);
    scores1 = zeros(101, length(idx));
    for i = 1 : length(idx)
        fid = idx(i);
        imgs = prepare_image_vgg(frames.imgs(:,:,:,fid), IMG_DIM);
        input_data = {imgs};
        s = caffe('forward', input_data);
        s = squeeze(s{1});
        S(:, (i-1) * 10 + 1 : i * 10) = squeeze(s);
        % Classifies the video.
        X_fc7 = max(0, max(0, s') * W_fc7 + B_fc7);
        X_fc8 = max(0, X_fc7 * W_fc8 + B_fc8);
        prob = exp(X_fc8);
        prob = exp(X_fc8) ./ repmat(sum(prob, 2)+eps, 1, 101);
        scores1(:, i) = mean(prob', 2);
    end
    S = S';
    save([out_path filesep num2str(vid) '_fc6.mat'], 'S');
    
    scores{vid} = scores1;
    [~, a] = max(scores1);
    b = accumarray(a(:), 1);
    [c, pred] = max(b);
    fprintf('Split %d, VIDEO %d: label = %d, pred = %d \n', ...
        split, vid, class_labels(vid), pred);
end

S = scores;
save([score_path filesep 'ucf101augVGG16Split' ...
    num2str(split) '_iter100000_scores'], 'S');


