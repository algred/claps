fid = fopen(['data' filesep 'ucf101_split' num2str(split) '_train2.txt'], 'r');
X = textscan(fid, '%s %d\n', -1);
fnames = X{1};
cidx = X{2};
fclose(fid);

fid = fopen(['data' filesep 'ucf101_split' num2str(split) '_val2.txt'], 'r');
X = textscan(fid, '%s %d\n', -1);
fnames_val = X{1};
cidx_val = X{2};
fclose(fid);

all_fnames = [fnames; fnames_val];
all_cidx = [cidx; cidx_val];
rnd_idx = randperm(length(all_cidx));
all_fnames2 = all_fnames(rnd_idx);
all_cidx2 = all_cidx(rnd_idx);

fid = fopen(['data' filesep 'ucf101_split' num2str(split) '_train_all.txt'], 'w');
for i = 1:length(all_cidx2)
    fprintf(fid, '%s %d\n', all_fnames2{i}, all_cidx2(i));
end
fclose(fid);
