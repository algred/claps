load('loc/trainValImgIdx.mat');
imgnet_fnames = [];
ind = 1;
idx = randperm(length(trainImgIdx));
N = 50000;
for i = 1:length(idx)
    try
        id = idx(i);
        im = imread(trainImgIdx(id).path);
    catch
        continue;
    end
    imgnet_fnames{ind} = trainImgIdx(id).path;
    ind = ind + 1;
    if ind > N 
        break;
    end
end 
save('imgnet_imgs_select.mat', 'imgnet_fnames');
