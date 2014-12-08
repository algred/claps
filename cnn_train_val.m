init_ucf101;
K = 1;
N = 30;
out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);

% Splits the training videos into training and validation set.
if ~exist('train_val_split.mat', 'file')
    train_idx = [];
    val_idx = [];
    for cid = 1:101
        idx = find((class_labels == cid) & (used_for_testing ~= 1));
        idx = idx(:);
        gid = [video_list(idx).group_id];
        gid = gid(:);
        unique_gid = unique(gid);
        val_group_num = max(1, round(length(unique_gid) / 4));
        val_gids = unique_gid(randsample(length(unique_gid), val_group_num));
        train_gids = setdiff(unique_gid, val_gids);
        for i = 1:length(val_gids)
            val_idx = [val_idx; idx(gid == val_gids(i))];
        end
        for i = 1:length(train_gids)
            train_idx = [train_idx; idx(gid == train_gids(i))];
        end
    end
    save('train_val_split.mat', 'train_idx', 'val_idx');
else
    load('train_val_split.mat');
end

% Samples N video frames from each training video.
train_imgs = cell(1, length(train_idx));
parfor i = 1:length(train_idx)
    vid = train_idx(i);
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4) - K;
    if nfms > N
        idx = randsample(nfms, N);
    else
        idx = 1:nfms;
    end
    train_imgs1 = cell(length(idx), 1);
    for j = 1:length(idx)
        id = idx(j) + K - 1;
        imname = [out_path filesep num2str(vid) filesep num2str(id) '.png'];
        train_imgs1{j} = sprintf('%s %d\n', imname, class_labels(vid) - 1);
    end
    train_imgs{i} = train_imgs1;
end
tmp = cell(1, N * length(train_idx));
ind = 1;
for i = 1:length(train_imgs)
    for j = 1:length(train_imgs{i})
        tmp{ind} = train_imgs{i}{j};
        ind = ind + 1;
    end
end
train_imgs = tmp;
if ind <= length(train_imgs)
    train_imgs(ind + 1 : end) = [];
end

val_imgs = cell(1, length(val_idx));
parfor i = 1:length(val_idx)
    vid = val_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4) - K;
    if nfms > N
        idx = randsample(nfms, N);
    else
        idx = 1:nfms;
    end
    val_imgs1 = cell(length(idx), 1);
    for j = 1:length(idx)
        id = idx(j) + K - 1;
        imname = [out_path filesep num2str(vid) filesep num2str(id) '.png'];
        val_imgs1{j} = sprintf('%s %d\n', imname, class_labels(vid) - 1);
    end
    val_imgs{i} = val_imgs1;
end
tmp = cell(1, N * length(val_idx));
ind = 1;
for i = 1:length(val_imgs)
    for j = 1:length(val_imgs{i})
        tmp{ind} = val_imgs{i}{j};
        ind = ind + 1;
    end
end
val_imgs = tmp;
if ind <= length(train_imgs)
    val_imgs(ind + 1 : end) = [];
end

% Writes out the lists of images.
fid = fopen(['ucf101_K' num2str(K) '_train.txt'], 'w');
idx = randperm(length(train_imgs));
for i = 1:length(idx)
    fprintf(fid, '%s',  train_imgs{idx(i)});
end
fclose(fid);

fid = fopen(['ucf101_K' num2str(K) '_val.txt'], 'w');
idx = randperm(length(val_imgs));
for i = 1:length(idx)
    fprintf(fid, '%s',  val_imgs{idx(i)});
end
fclose(fid);

