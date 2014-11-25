init_ucf101;

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
out_path = pathstring('/research/action_data/ucf101-singleframes');
train_imgs = cell(1, 200 * length(train_idx));
ind = 1;
for i = 1:length(train_idx)
    vid = train_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    load([frame_path filesep num2str(vid) '_frames.mat']);
    for j = 1:size(imgs, 4)
        imname = [out_path filesep num2str(vid) filesep num2str(j) '.png'];
        imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
        train_imgs{ind} = sprintf('%s %d\n', imname, class_labels(vid));
        ind = ind + 1;
    end
end
train_imgs(ind + 1 : end) = [];

val_imgs = cell(1, 200 * length(val_idx));
ind = 1;
for i = 1:length(val_idx)
    vid = val_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    load([frame_path filesep num2str(vid) '_frames.mat']);
    for j = 1:size(imgs, 4)
        imname = [out_path filesep num2str(vid) filesep num2str(j) '.png'];
        imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
        val_imgs{ind} = sprintf('%s %d\n', imname, class_labels(vid));
        ind = ind + 1;
    end
end
val_imgs(ind + 1 : end) = [];

% Writes out the lists of images.
fid = fopen('ucf101_singleframe_train.txt', 'w');
idx = randperm(length(train_imgs));
for i = 1:length(idx)
    fprintf(fid, '%s',  train_imgs{idx(i)});
end
fclose(fid);

fid = fopen('ucf101_singleframe_val.txt', 'w');
idx = randperm(length(val_imgs));
for i = 1:length(idx)
    fprintf(fid, '%s',  val_imgs{idx(i)});
end
fclose(fid);

