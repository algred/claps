function im = transform_image(im)
im = im(:, :, [3 2 1]);
im = permute(im, [2 1 3]);
