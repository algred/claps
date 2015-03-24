% addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab'));
init_ucf101;

use_gpu = 1;
%% Model files for VGG 16 layer model.
% model_path = ['/research/action_videos/shared/caffe-dev/models/ucf101_very_deep'];
% model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_path = ['/research/action_videos/caffe_v1.0/caffe/models/ucf101_very_deep'];
model_def_file = ['caffe' filesep 'verydeep_deploy.prototxt']
model_file = [model_path filesep 'ucf101augVGG16K1AllM52_iter_80000.caffemodel']
K = 1;

%% Model files for oxford 2048M model.
% model_path = ['/research/action_videos/shared/caffe-dev/models/ucf101aug_K' num2str(K)];
% model_file = [model_path filesep 'ucf101AUGK1All_iter_80000.caffemodel'];
% model_def_file = [model_path filesep 'deploy.prototxt']

matcaffe_init(use_gpu, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
N = 20;
for i = 1:length(video_list)
    if used_for_testing(i) ~= 1
        continue;
    end
    
    load([frame_path filesep num2str(i) '_frames.mat']);
    nfms = size(imgs, 4);
    delta = max(1, floor(nfms / N));
    if N >= nfms
        N1 = nfms;
    else
        N1 = N;
    end
    scores = zeros(101, N1);
    
    for j = 1:N1
        input_data = {prepare_image(imgs(:, :, :, (j-1)*delta + 1), 224)};
        s = caffe('forward', input_data);
        scores(:, j) = mean(squeeze(s{1}), 2);
    end
    [~, a] = max(scores);
    S{i} = scores;
    b = accumarray(a(:), 1);
    [c, pred(i)] = max(b);
    fprintf('VIDEO %d: label = %d, pred = %d \n', i, class_labels(i), pred(i));
end
save([out_path filesep 'verydeep_augK' num2str(K) ...
    'all_modified52_iter80000_scores_test_0322.mat'], 'S');

