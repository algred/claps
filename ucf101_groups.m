load('ucf101_data.mat');

for i = 1:length(video_list)
    video_list2(i).video_name = video_list(i).video_name;
    idx = strfind(video_list(i).video_name, '_');
    gid = str2num(video_list(i).video_name(idx(end-1)+2 : idx(end)-1));
    video_list2(i).group_id = gid;
end