function images = prepare_image_vgg(im, cropped_dim)
% ------------------------------------------------------------------------
IMAGE_DIM = 256;

if exist('cropped_dim', 'var')
    CROPPED_DIM = cropped_dim;
else
    CROPPED_DIM = 227;
end

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR
im = im(:,:,[3 2 1]);
im(:,:,1) = im(:,:,1) - 103.939;
im(:,:,2) = im(:,:,2) - 116.779;
im(:,:,3) = im(:,:,3) - 123.68;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
