addpath(genpath('/research/action_videos/shared/caffe-dev/matlab'));
init_ucf101;

model_path = ['/research/action_videos/video_data/deepnet_ucf101/caffemodels'];
model_def_file = ['caffe' filesep 'verydeep_deploy.prototxt']
model_file = [model_path filesep 'ucf101augVGG16S4All_iter_40000.caffemodel']
matcaffe_init(1, model_def_file, model_file);

out_path = '/research/action_videos/video_data/deepnet_ucf101';
pred = zeros(length(video_list), 1);
S = cell(length(video_list), 1);
N = 10;
K = 2;
sqK = K * K;
IMG_DIM = 224;
SUB_IMG_DIM = IMG_DIM / K;
BZ = 10;

for vid = 1:length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4);
    intv = floor(nfms / sqK);
    imgs = zeros(IMG_DIM, IMG_DIM, 3, N, 'single');
    IX = rand_idx(intv, sqK, N, nfms);
    for i = 1:N
        ix = IX(i, :);
        % Makes the image.
        for j = 1:K
            for h = 1:K
                id = (j - 1) * K + h;
                subim = single(imresize(frames.imgs(:, :, :, ix(id)), ...
                    [SUB_IMG_DIM SUB_IMG_DIM], 'bilinear'));
                imgs((j-1) * SUB_IMG_DIM + 1 : j * SUB_IMG_DIM, ...
                    (h-1) * SUB_IMG_DIM + 1 : h * SUB_IMG_DIM, :, i) = ...
                    transform_image(subim);
            end
        end
    end

    imgs(:, :, 1, :) = imgs(:, :, 1, :) - 103.939;
    imgs(:, :, 2, :) = imgs(:, :, 2, :) - 116.779;
    imgs(:, :, 3, :) = imgs(:, :, 3, :) - 123.68;

    scores = zeros(101, N);
    for i = 1:BZ:N
        input_data = {imgs(:,:,:,i:i+BZ-1)};
        s = caffe('forward', input_data);
        scores(:, i:i+BZ-1) = squeeze(s{1});
    end
    S{vid} = scores;
    [max_score, pred(vid)] = max(mean(scores, 2), [], 1);
    fprintf('VIDEO %d: label = %d, pred = %d \n', ...
        vid, class_labels(vid), pred(vid));
end
save([out_path filesep 'verydeep_augVGG16S' num2str(sqK) ...
    'All_iter40000_scores_0311.mat'], 'S');

