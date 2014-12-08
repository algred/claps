function imgs2 = clap_frames(imgs, M, K)
% Claps K consecutive frames given an array of images and foreground masks M.
rows = size(imgs, 1);
cols = size(imgs, 2);
nfms = size(M, 4);
imgs2 = zeros(rows, cols, 3, nfms - (K - 1), 'uint8');
imgs = double(imgs);
for i = K : nfms
    intv_idx = (i - K + 1) : i;
    fg_img = sum(imgs(:, :, :, intv_idx) .* M(:, :, :, intv_idx), 4);
    W = sum(M(:, :, :, intv_idx), 4);
    im = imgs(:, :, :, i) .* (W <= 0) + fg_img .* (W > 0) ./ (W + eps);
    imgs2(:, :, :, i - K + 1) = uint8(im);
end
end
