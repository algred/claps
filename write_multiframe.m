init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
K = 10;

% Splits the training videos into training and validation set.
if ~exist('single_frame_train_val.mat', 'file')
    idx  = find(used_for_testing ~= 1);
    train_idx = [];
    val_idx = [];
    for c = 1 : length(class_names)
        idx1 = idx(class_labels(idx) == c);
        n = length(idx1);
        train_idx1 = idx1(randsample(n, floor(n * (3 / 4))));
        val_idx1 = setdiff(idx1, train_idx1);
        train_idx = [train_idx; train_idx1(:)];
        val_idx = [val_idx; val_idx1(:)];
    end
    save('single_frame_train_val.mat', 'train_idx', 'val_idx');
else
    load('single_frame_train_val.mat');
end

% Writes out the video frames.
out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);
parfor i = 1:length(train_idx)
    vid = train_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    flow = load([flow_path filesep num2str(vid) '_flow.mat']);
    flow = decompress_flow(flow.flow_int, flow.flow_frac);
    imgs = clap_frames(flow, frames.imgs, K);
    
    for j = 1:size(imgs, 4)
        imname = [out_path filesep num2str(vid) filesep num2str(j + K - 1) '.png'];
        imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
    end
end

parfor i = 1:length(val_idx)
    vid = val_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    flow = load([flow_path filesep num2str(vid) '_flow.mat']);
    flow = decompress_flow(flow.flow_int, flow.flow_frac);
    imgs = clap_frames(flow, frames.imgs, K);
    
    for j = 1:size(imgs, 4)
        imname = [out_path filesep num2str(vid) filesep num2str(j + K - 1) '.png'];
        imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
    end
end

