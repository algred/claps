init_ucf101;
img_root = pathstring(['/research/action_data/ucf101-S4']);

fid = fopen(['data' filesep 'ucf101_S4_test.txt'], 'w');
for vid = 1:length(video_list)
    if used_for_testing(vid) ~= 1
        continue;
    end
    img_dir = [img_root filesep num2str(vid)];
    img_list = dir([img_dir filesep '*.png']);
    for i = 1:length(img_list)
        fprintf(fid, '%s %d\n', [img_dir filesep img_list(i).name], ...
            class_labels(vid) - 1);
    end
end
fclose(fid);

