vframe = load(['data' filesep 'vframe_vis_sample.mat']);
webimg = load(['data' filesep 'webimg_vis_sample.mat']);
imgnet = load('imgnet_imgs_select.mat');
img_names = [vframe.img_names(:); webimg.img_names(:); imgnet.imgnet_fnames(:)];
source_id = [ones(length(vframe.img_names), 1); ...
    ones(length(webimg.img_names), 1) * 2; ...
    ones(length(imgnet.imgnet_fnames), 1) * 3];

layer_name = 'fc6';
net_name = 'ucf101vgg16K1';
data_path = '/research/action_videos/video_data/deepnet_ucf101/'
A1 = load([data_path filesep 'visual_data' filesep ...
    'activation_vframe_' net_name '_' layer_name]);
A2 = load([data_path filesep 'visual_data' filesep ...
    'activation_webimg_' net_name '_' layer_name]);
A3 = load([data_path filesep 'visual_data' filesep ...
    'activation_imgnet_' net_name '_' layer_name]);
S = [A1.A; A2.A; A3.A];
[S_sorted, IX] = sort(S, 'descend');

im_file = ['visualize' filesep net_name filesep 'all' filesep layer_name '.txt'];
% fid = fopen(im_file, 'w');

K = 10;
N = size(IX, 2);
bad_idx = [];
for i = 1:N
    idx = IX(:, i);
    vid_map = containers.Map;
    ind = 1;
    for j = 1:1000
        if ind > K
            continue;
        end
        id = idx(j);
        sid = source_id(id);
        if sid == 1
            imname = img_names{id};
            a = strfind(imname, '/');
            vid = imname(a(end-1)+1 : a(end) - 1);
            if isKey(vid_map, vid)
                continue;
            end
            vid_map(vid) = ind;
        end
        if sid == 2
            im1 = imread(pathstring(img_names{id}));
            if ndims(im1) < 3
                bad_idx = [bad_idx; id];
                continue;
            end
            im1_type = whos('im1');
            if strcmp(im1_type.class, 'uint16')
                bad_idx =  [bad_idx; id];
                continue;
            end
        end
        % fprintf(fid, '%s %d %d\n', pathstring(img_names{id}), i, sid);
        % ind = ind + 1;
    end
end

