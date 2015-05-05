% addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab'));
init_ucf101;

use_gpu = 1;
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels/ucf'];
model_def_file = ['caffe' filesep 'VGG_CNN_M_2048_deploy.prototxt'];
model_file = [model_path filesep 'ucf101VGG7Split' num2str(split)...
    '_iter_80000.caffemodel'];

matcaffe_init(use_gpu, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
N = 20;
for i = 1:length(video_list)
    if used_for_testing(i) ~= split
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
        input_data = {prepare_image_vgg(imgs(:, :, :, (j-1)*delta + 1), 224)};
        s = caffe('forward', input_data);
        scores(:, j) = mean(squeeze(s{1}), 2);
    end
    [~, a] = max(scores);
    S{i} = scores;
    b = accumarray(a(:), 1);
    [c, pred(i)] = max(b);
    fprintf('VIDEO %d: label = %d, pred = %d \n', i, class_labels(i), pred(i));
end
save([out_path filesep 'ucf101VGG7Split' num2str(split) ...
    '_iter80000_scores_test.mat'], 'S');

