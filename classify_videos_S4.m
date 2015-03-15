addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
% init_ucf101;
frame_path = '/research/action_data/ucf101-frames/';
load('ucf101_data.mat');

use_gpu = 1;
model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = ['caffe' filesep 'verydeep_deploy.prototxt']
% model_file = [model_path filesep 'ucf101augVGG16K1All_iter_80000.caffemodel']
model_file = [model_path filesep 'ucf101augVGG16S4All_iter_40000.caffemodel']
matcaffe_init(use_gpu, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
N = 10;
K = 2;
sqK = K * K;
IMG_DIM = 224;
SUB_IMG_DIM = round(IMG_DIM / K);
BZ = 10;

video_batch = 10;
for vid = 1:length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4);
    intv = floor(nfms / sqK);
    vid2 = mod(vid, video_batch) + 1;
    imgs{vid2} = zeros(IMG_DIM, IMG_DIM, 3, N, 'single');
    %IX = rand_idx(intv, sqK, N, nfms);
    intv2 = floor(intv / 2);
    IX1 = [intv2 intv+intv2 intv*2+intv2 intv*3+intv2];
    IX1 = [IX1; IX1-2; IX1+2];
    ind = 1;
    IX = [];
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for h = 1:3
                    IX(ind, :) = [IX1(i, 1) IX1(j, 2) IX1(k, 3) IX1(h, 4)];
                    ind = ind + 1;
                end
            end
        end
    end
    IX = IX(1:N, :);

    for i = 1:N
        ix = IX(i, :);
        % Makes the image.
        for j = 1:K
            for h = 1:K
                id = (j - 1) * K + h;
                subim = imresize(single(frames.imgs(:, :, :, ix(id))), ...
                    [SUB_IMG_DIM SUB_IMG_DIM], 'bilinear');
                subim = transform_image(subim);
                subim(:,:,1) = subim(:,:,1) - single(103.939);
                subim(:,:,2) = subim(:,:,2) - single(116.779);
                subim(:,:,3) = subim(:,:,3) - single(123.68);

                imgs{vid2}((j-1) * SUB_IMG_DIM + 1 : j * SUB_IMG_DIM, ...
                    (h-1) * SUB_IMG_DIM + 1 : h * SUB_IMG_DIM, :, i) = subim;
            end
        end
    end
    if vid2 < video_batch && vid < length(video_list)
        continue;
    end
    for ii = 1:video_batch
        if isempty(imgs{ii})
            continue;
        end
        scores = zeros(101, N);
        for i = 1:BZ:N
           input_data = {imgs{ii}(:,:,:,i:i+BZ-1)};
           s = caffe('forward', input_data);
           scores(:, i:i+BZ-1) = squeeze(s{1});
        end
        this_vid = vid - (video_batch - ii); 
        S{this_vid} = scores;
        [max_score, pred(this_vid)] = max(mean(scores, 2), [], 1);
        fprintf('VIDEO %d: label = %d, pred = %d \n', ...
            vid, class_labels(this_vid), pred(this_vid));
    end
end
save([out_path filesep 'verydeep_augVGG16S' num2str(sqK) ...
    'All_iter40000_scores_0312.mat'], 'S');

