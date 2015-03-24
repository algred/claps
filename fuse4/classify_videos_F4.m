addpath(genpath('/research/action_videos/caffe_v1.0/caffe/matlab/caffe'));
addpath('..')
frame_path = '/research/action_data/ucf101-frames/';
load('../ucf101_data.mat');

% model_path = '/research/action_videos/caffe_v1.0/caffe/models/ucf101_vgg16_fuse4';
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = ['deploy.prototxt']
% model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
model_file = [model_path filesep 'ucf101augVGG16Fuse4V2_iter_40000.caffemodel']
matcaffe_init(1, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
D = [0 0 0 0; 2 2 2 2; -2 -2 -2 -2; 2 2 -2 -2; ...
    -2 -2 2 2; -2 2 2 -2; 2 -2 -2 2];
N = size(D, 1);
IMG_DIM = 224;
for vid = 1:length(video_list)
    if used_for_testing(vid) ~= 1
        continue;
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4);
    intv = floor(nfms / 4);
    intv2 = floor(intv / 2);
    IX = [intv2 intv+intv2 intv*2+intv2 intv*3+intv2];
    IX = max(min(repmat(IX, N, 1) - D, nfms), 1); 
    scores = zeros(101, N);
    for i = 1:N
        for k = 1:4
            all_imgs{k} = prepare_image_vgg(frames.imgs(:,:,:,IX(i,k)), IMG_DIM);
        end
        bz = size(all_imgs{k}, 4);
        scores1 = zeros(101, bz);
        for h = 1:bz
            imgs = zeros(IMG_DIM, IMG_DIM, 12, 'single');
            for k = 1:4
                imgs(:,:,(k-1)*3 + 1) = all_imgs{k}(:,:,1,h);
                imgs(:,:,(k-1)*3 + 2) = all_imgs{k}(:,:,2,h);
                imgs(:,:,(k-1)*3 + 3) = all_imgs{k}(:,:,3,h);
            end
            input_data = {imgs};
            s = caffe('forward', input_data);
            scores1(:, h) = squeeze(s{1});
        end
        scores(:, i) = mean(scores1, 2); 
    end
    S{vid} = scores;
    [max_score, pred(vid)] = max(mean(scores, 2), [], 1);
    fprintf('VIDEO %d: label = %d, pred = %d \n', ...
        vid, class_labels(vid), pred(vid));
end
save([out_path filesep 'ucf101_AUG_VGG16_F4V2_iter40000_scores_0317.mat'], 'S');

