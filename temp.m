imgnet = load('imgnet_imgs_select.mat');
imgnet_flg = ones(length(imgnet.imgnet_fnames), 1);
parfor i = 1:length(imgnet.imgnet_fnames)
    im = imread(imgnet.imgnet_fnames{i});
    if ndims(im) < 3 || max(im(:)) > 255
        imgnet_flg(i) = 0;
    end
end
save('data/imgnet_flg.mat', 'imgnet_flg');
