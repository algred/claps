init_ucf101;

% % Splits the training videos into training and validation set.
% if ~exist('single_frame_train_val.mat', 'file')
%     idx  = find(used_for_testing ~= 1);
%     train_idx = [];
%     val_idx = [];
%     for c = 1 : length(class_names)
%         idx1 = idx(class_labels(idx) == c);
%         n = length(idx1);
%         train_idx1 = idx1(randsample(n, floor(n * (3 / 4))));
%         val_idx1 = setdiff(idx1, train_idx1);
%         train_idx = [train_idx; train_idx1(:)];
%         val_idx = [val_idx; val_idx1(:)];
%     end
%     save('single_frame_train_val.mat', 'train_idx', 'val_idx');
% else
%     load('single_frame_train_val.mat');
% end

% Writes out the video frames.
out_path = pathstring('/research/action_data/ucf101-k1');
test_idx = find(used_for_testing == 1);
parfor i = 1:length(test_idx)
    vid = test_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    for j = 1:size(frames.imgs, 4)
        imname = [out_path filesep num2str(vid) filesep num2str(j) '.png'];
        imwrite(imresize(frames.imgs(:, :, :, j), [256, 256]), imname, 'png');
    end
end

% for i = 1:length(val_idx)
%     vid = val_idx(i);
%     if ~exist([out_path filesep num2str(vid)], 'file')
%         mkdir([out_path filesep num2str(vid)]);
%     end
%     load([frame_path filesep num2str(vid) '_frames.mat']);
%     for j = 1:size(imgs, 4)
%         imname = [out_path filesep num2str(vid) filesep num2str(j) '.png'];
%         imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
%     end
% end

