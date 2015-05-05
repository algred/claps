addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab/caffe'));
init_ucf101;
score_path = '/research/action_videos/video_data/deepnet_ucf101';

% Reads the weights of fc7, fc8.
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels/ucf'];

weights_file = [model_path filesep 'ucf101augVGG16Split' ...
    num2str(split) '_iter_100000_weights.mat']
if exist(weights_file, 'file')
    load(weights_file);
else
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
    save(weights_file,'W_fc7', 'B_fc7', 'W_fc8', 'B_fc8');
end

feat_path = pathstring(['X:\video_data\deepnet_ucf101\cnnfeat\split' ...
    num2str(split) filesep 'fc6'])

IMG_DIM = 224;
FEAT_DIM = 4096;
N = 20;
scores = cell(length(video_list), 1);
for vid = 1:length(video_list)
    if used_for_testing(vid) ~= split
        continue;
    end
    load([feat_path filesep num2str(vid) '_fc6.mat']);
    scores1 = zeros(101, floor(size(S, 1) / 10));
    ind = 1;
    for i = 1 : 10 : size(S, 1)
        X_fc7 = max(0, max(0, S(i:i+9, :)) * W_fc7 + B_fc7);
        X_fc8 = max(0, X_fc7 * W_fc8 + B_fc8);
        prob = exp(X_fc8);
        prob = exp(X_fc8) ./ repmat(sum(prob, 2)+eps, 1, 101);
        scores1(:, ind) = mean(prob', 2);
        ind = ind + 1;
    end
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

