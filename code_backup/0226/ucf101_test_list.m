init_ucf101;
K = 1;
N = 30;
out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);

% Samples N video frames from each testing video.
test_idx = find(used_for_testing == 1);
img_list = cell(1, length(test_idx));
for i = 1:length(test_idx)
    vid = test_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        fprintf('ERROR: frames for video %d not found.\n', vid);
        continue;
    end
    frame_list = dir([out_path filesep num2str(vid) filesep '*.png']);
    nfms = length(frame_list);
    if nfms > N
        idx = randsample(nfms, N);
    else
        tt = floor(N / nfms);
        idx = repmat(1:nfms, tt, 1);
        idx = [idx(:); randsample(nfms, N - nfms * tt)];
    end
    img_list1 = cell(length(idx), 1);
    for j = 1:length(idx)
        id = idx(j) + K - 1;
        imname = [out_path filesep num2str(vid) filesep num2str(id) '.png'];
        img_list1{j} = sprintf('%s %d\n', imname, class_labels(vid) - 1);
    end
    img_list{i} = img_list1;
end
tmp = cell(1, N * length(test_idx));
ind = 1;
for i = 1:length(img_list)
    for j = 1:length(img_list{i})
        tmp{ind} = img_list{i}{j};
        ind = ind + 1;
    end
end
img_list = tmp;
if ind <= length(img_list)
    img_list(ind + 1 : end) = [];
end

% Writes out the lists of images.
fid = fopen(['ucf101_K' num2str(K) '_test.txt'], 'w');
idx = randperm(length(img_list));
for i = 1:length(idx)
    fprintf(fid, '%s',  img_list{idx(i)});
end
fclose(fid);

