function images = prepare_image(im)
% ------------------------------------------------------------------------
d = load('/research/action_videos/shared/caffe-dev/models/vgg_cnn/VGG_mean');
IMAGE_DIM = 256;
CROPPED_DIM = 224;
IMAGE_MEAN = d.image_mean;

% resize to fixed input size
im = single(im);

if size(im, 1) < size(im, 2)
    im = imresize(im, [IMAGE_DIM NaN]);
else
    im = imresize(im, [NaN IMAGE_DIM]);
end

% RGB -> BGR
im = im(:, :, [3 2 1]);

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');

indices_y = [0 size(im,1)-CROPPED_DIM] + 1;
indices_x = [0 size(im,2)-CROPPED_DIM] + 1;
center_y = floor(indices_y(2) / 2)+1;
center_x = floor(indices_x(2) / 2)+1;

curr = 1;
for i = indices_y
  for j = indices_x
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)-IMAGE_MEAN, [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
images(:,:,:,5) = ...
    permute(im(center_y:center_y+CROPPED_DIM-1,center_x:center_x+CROPPED_DIM-1,:)-IMAGE_MEAN, ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);

