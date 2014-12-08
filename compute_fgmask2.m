function M = compute_fgmask2(imgs, hu, hv)
% Batch compute foreground masks for an array of images. 
% Note the mask for an image has 3 channels of duplicate values.

[rows, cols, ~, nfms] = size(imgs);
nfms = nfms - 1;
spx = zeros(rows, cols, nfms);
for i = 1:nfms
    spx(:, :, i) = vl_slic(single(vl_rgb2xyz(imgs(:, :, :, i))), 30, 1);
end

M = zeros(rows, cols, 3, nfms);
for i = 1:nfms
    motion = sqrt(hu(:, :, i) .* hu(:, :, i) + hv(:, :, i) .* hv(:, :, i));
    t = kmeans2cls(motion);
    spx1 = spx(:, :, i);
    
    m = false(rows, cols);
    for k = 1:max(spx1(:))
        m(spx1(:) == k) = (mean(motion(spx1(:) == k)) > t);
    end
    M(:, :, :, i) = repmat(bwmorph(m, 'close', inf), [1 1 3]);
end