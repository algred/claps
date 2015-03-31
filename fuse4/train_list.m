split = 1;

train_splits = setdiff([0:3], split);
X = [];
for i = 1:length(train_splits)
    split1 = train_splits(i);
    fid = fopen(['../data/ucf101_split' num2str(split1) '_hdf5.txt'], 'r');
    X1 = textscan(fid, '%s\n', -1);
    X = [X; X1{1}];
    fclose(fid);
end

fid = fopen('../data/ucf101img_hdf5_list.txt', 'r');
X1 = textscan(fid, '%s\n', -1);
X = [X; X1{1}];
fclose(fid);

idx = randperm(length(X));
fid = fopen(['../data/ucf101_fuse4_train_split' num2str(split) '_v2.txt'], 'w');
for i = 1:length(X)
    fprintf(fid, '%s\n', X{idx(i)});
end
fclose(fid);
