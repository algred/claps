split = 1;
addpath('..');
init_ucf101;
data_root = '/scratch/shugao/ucf101-flow-hdf5';

N_train = sum(used_for_testing ~= split);
N_test = length(video_list) - N_train;
train_file_list = cell(N_train, 1);
test_file_list = cell(N_test, 1);
train_ind = 1;
test_ind = 1;
for vid = 1:length(video_list)
    file_list = dir([data_root filesep num2str(vid) filesep '*.h5']);
    if used_for_testing(vid) == split
        for i = 1:length(file_list)
            test_file_list{test_ind} = [data_root filesep ...
                num2str(vid) filesep file_list(i).name];
            test_ind = test_ind + 1;
        end
    else
        for i = 1:length(file_list)
            train_file_list{train_ind} = [data_root filesep ...
                num2str(vid) filesep file_list(i).name];
            train_ind = train_ind + 1;
        end
    end
end

idx = randperm(length(test_file_list));
fid = fopen(['ucf101_flow_test_split' num2str(split) '.txt'], 'w');
for i = 1:length(idx)
    id = idx(i);
    fprintf(fid, '%s\n', test_file_list{id});
end
fclose(fid);

idx = randperm(length(train_file_list));
fid = fopen(['ucf101_flow_train_split' num2str(split) '.txt'], 'w');
for i = 1:length(idx)
    id = idx(i);
    fprintf(fid, '%s\n', train_file_list{id});
end
fclose(fid);
