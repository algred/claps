split = 3;
phase = 'train'

fid = fopen(['data' filesep 'ucf101_split' num2str(split) '_' phase '.txt'], 'r');
X = textscan(fid, '%s %d\n', -1);
fnames = X{1};
cidx = X{2};
fclose(fid);

for i = 1:length(fnames)
    idx = strfind(fnames{i}, '/');
    vid(i) = str2num(fnames{i}(idx(end-1)+1 : idx(end)-1));
end

N = 30;
unique_vid = unique(vid);
sidx = [];
for i = 1:length(unique_vid)
    idx = find(vid == unique_vid(i));
    idx = idx(randsample(length(idx), 30));
    sidx = [sidx; idx(:)];
end

sidx = sort(sidx);
fid = fopen(['data' filesep 'ucf101_split' num2str(split) '_' phase '2.txt'], 'w');
for i = 1:length(sidx)
    id = sidx(i);
    fprintf(fid, '%s %d\n', fnames{id}, cidx(id));
end
fclose(fid);
