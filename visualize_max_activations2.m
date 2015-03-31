vframe = load(['data' filesep 'vframe_vis_sample.mat']);
webimg = load(['data' filesep 'webimg_vis_sample.mat']);
imgnet = load('imgnet_imgs_select.mat');
img_names = [vframe.img_names(:); webimg.img_names(:); imgnet.imgnet_fnames(:)];
source_id = [ones(length(vframe.img_names), 1); ...
    ones(length(webimg.img_names), 1) * 2; ...
    ones(length(imgnet.imgnet_fnames), 1) * 3];

layer_name = 'conv5_2';
rf = 175;
net_name = 'ucf101vgg16K1';
nchs = 512;
stride = 16;
map_dim = 14;

A1 = load(['visual_data' filesep 'activation_vframe_' net_name '_' layer_name]);
A2 = load(['visual_data' filesep 'activation_webimg_' net_name '_' layer_name]);
A3 = load(['visual_data' filesep 'activation_imgnet_' net_name '_' layer_name]);
A = [A1.A; A2.A; A3.A];

img_dim = 224;
hrf = floor(rf / 2);

rfim_path = ['visualize' filesep net_name filesep 'all' filesep layer_name];

if ~exist(rfim_path, 'file')
    mkdir(rfim_path);
end

S = A(:, 1:nchs);
[S_sorted, IX] = sort(S, 'descend');

K = 10;
N = size(IX, 1);
for i = 1:nchs
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
        im1 = imread(pathstring(img_names{id}));
        if ndims(im1) < 3
            continue;
        end
        im1_type = whos('im1');
        if strcmp(im1_type.class, 'uint16')
            continue;
        else
            im = zeros(img_dim + hrf * 2, img_dim + hrf * 2, 3, 'uint8');
        end
        im1 = imresize(im1, [img_dim, img_dim], 'bilinear');
        im(hrf + 1 : hrf + img_dim, hrf + 1 :  hrf + img_dim, :) = im1;
        uid = A(id, nchs + i);
        [ur, uc] = ind2sub([map_dim, map_dim], uid);
        x = (uc - 1) * stride + 1;
        y = (ur - 1) * stride + 1;
        rfim = im(x : x + rf - 1, y : y + rf - 1, :);
        imwrite(rfim, [rfim_path filesep num2str(i) '_' num2str(ind) ...
            '_' num2str(sid) '.png'], 'png'); 
        ind = ind + 1;
    end
end

