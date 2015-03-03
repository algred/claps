fid = fopen(['data' filesep 'ucf101_K1_train.txt'], 'r');
X = textscan(fid, '%s %d\n', -1);
fnames = X{1};
cidx = X{2};
fclose(fid);

fid = fopen(['data' filesep 'ucf101img_aug.txt'], 'r');
X = textscan(fid, '%s %d\n', -1);
aug_fnames = X{1};
aug_cidx = X{2};
fclose(fid);

N = 100; 
T = 20;
idx = cell(101, 1);
for i = 1:101
    idx{i} = find(aug_cidx == (i - 1));
    idx{i} = idx{i}(:);
end
sampled_idx = cell(101, 1);
for i = 1:T
    for j = 1:101
        sampled_idx{j} = [sampled_idx{j}; idx{j}(randsample(length(idx{j}), N))];
    end
end
sampled_idx = cell2mat(sampled_idx);

all_fnames = [fnames; aug_fnames(sampled_idx)];
all_cidx = [cidx; aug_cidx(sampled_idx)];
rnd_idx = randperm(length(all_cidx));
all_fnames2 = all_fnames(rnd_idx);
all_cidx2 = all_cidx(rnd_idx);

fid = fopen(['data' filesep 'ucf101aug_K1_train.txt'], 'w');
for i = 1:length(all_cidx2)
    fprintf(fid, '%s %d\n', all_fnames2{i}, all_cidx2(i));
end
fclose(fid);
