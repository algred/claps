init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
K = 5;
N = 30;

% Splits the training videos into training and validation set.
load('single_frame_train_val.mat');
out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);

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

