addpath(genpath('/research/action_videos/shared/caffe-master/matlab'));
init_ucf101;
K = 10;

use_gpu = 1;
model_path = ['/research/action_videos/shared/caffe-master/models/finetune_ucf101/K' num2str(K)];
model_def_file = [model_path filesep 'deploy.prototxt']
model_file = [model_path filesep 'finetune_ucf101_K' num2str(K) '_iter_2000.caffemodel'];
matcaffe_init(use_gpu, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
for i = 1:length(video_list)
    if used_for_testing(i) ~= 1
        continue;
    end
    load([frame_path filesep num2str(i) '_frames.mat']);
    scores = zeros(101, size(imgs, 4));
    for j = 1:size(imgs, 4)
        input_data = {prepare_image(imgs(:, :, :, j))};
        s = caffe('forward', input_data);
        scores(:, j) = mean(squeeze(s{1}), 2);
    end
    [~, a] = max(scores);
    S{i} = scores;
    b = accumarray(a(:), 1);
    [c, pred(i)] = max(b);
    fprintf('VIDEO %d: label = %d, pred = %d \n', i, class_labels(i), pred(i));
end
save([out_path filesep 'K' num2str(K) '_scores.mat'], 'S');
