addpath(genpath('/research/action_videos/shared/caffe-master/matlab'));
init_ucf101;

% init caffe network (spews logging info)
use_gpu = 1;
model_path = '/research/action_videos/shared/caffe-master/models/finetune_ucf101/singleframe';
model_def_file = [model_path filesep 'deploy.prototxt']
model_file = [model_path filesep 'finetune_ucf101_singleframe_iter_2000.caffemodel'];
matcaffe_init(use_gpu, model_def_file, model_file);

for i = 1:length(video_list)
    if used_for_testing(i) ~= 1
        continue;
    end
    load([frame_path filesep num2str(i) '_frames.mat']);
    % scores = cell(size(imgs, 4), 1);
    for j = 1:size(imgs, 4)
        input_data = prepare_image(imgs(:, :, :, j));
        scores = caffe('forward', input_data);
        keyboard;
    end
end


