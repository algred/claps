addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;
K = 10;
iter = 40000;

use_gpu = 1;
model_path = ['/research/action_videos/shared/caffe-dev/models/ucf101_K' num2str(K)];
model_def_file = [model_path filesep 'deploy.prototxt']
model_file = [model_path filesep 'ucf101K' num2str(K) ...
    'aug_iter_' num2str(iter) '.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);

frame_path = '/research/action_videos/video_data/ucf101_K10TR_frames';
out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
for i = 1:length(video_list)
    if used_for_testing(i) == 1
        continue;
    end
    load([frame_path filesep num2str(i) '_K' num2str(K) '_frames.mat']);
    scores = zeros(101, size(imgs, 4));
    for j = 1:size(imgs, 4)
        input_data = {prepare_image(imgs(:, :, :, j), 224)};
        s = caffe('forward', input_data);
        scores(:, j) = mean(squeeze(s{1}), 2);
    end
    [~, a] = max(scores);
    S{i} = scores;
    b = accumarray(a(:), 1);
    [c, pred(i)] = max(b);
    fprintf('VIDEO %d: label = %d, pred = %d \n', i, class_labels(i), pred(i));
end
save([out_path filesep 'oxford_aug_K' num2str(K) '_scores_train_0219.mat'], 'S');
