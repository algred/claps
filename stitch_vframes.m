init_ucf101;
out_root = pathstring(['/research/action_data/ucf101-S4']);

K = 2;
sqK = K * K;
N = 30;
IMG_DIM = 224;

% Assuming IMAGE_DIM is mod(IMAGE_DIM, K) = 0. 
SUB_IMG_DIM = IMG_DIM / K;

parfor vid = 1:length(video_list)
    if used_for_testing(vid) ~= 1
        continue;
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4);
    intv = floor(nfms / sqK);
    out_dir = [out_root filesep num2str(vid)];
    if ~exist(out_dir, 'file')
        mkdir(out_dir);
    end
    IX = rand_idx(intv, sqK, N, nfms);
    for i = 1:N
        ix = IX(i, :);
        % Composes the image name.
        img_name = sprintf('%03d', ix(1));
        for j = 2:sqK
            img_name = [img_name sprintf('-%03d', ix(j))];
        end
        img_name = [out_dir filesep img_name '.png'];
        % Makes the image.
        im = zeros(IMG_DIM, IMG_DIM, 3, 'uint8');
        for j = 1:K
            for h = 1:K
                id = (j - 1) * K + h;
                subim = imresize(frames.imgs(:, :, :, ix(id)), ...
                    [SUB_IMG_DIM SUB_IMG_DIM], 'bilinear');
                im((j-1) * SUB_IMG_DIM + 1 : j * SUB_IMG_DIM, ...
                    (h-1) * SUB_IMG_DIM + 1 : h * SUB_IMG_DIM, :) = subim;
            end
        end
        % Saves the image.
        imwrite(im, img_name, 'png');
    end
end

