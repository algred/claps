load('loc/trainValImgIdx.mat');
imgnet_fnames = [];
ind = 1;
for i = 1:length(trainImgIdx)
    try
        im = imread(trainImgIdx(i).path);
    catch
        continue;
    end
    imgnet_fnames{ind} = trainImgIdx(i).path;
    ind = ind + 1;
end 
save('imgnet_imgs.mat', 'imgnet_fnames');
