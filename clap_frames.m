function imgs2 = clap_frames(flow, imgs, K)
[rows, cols, ~, nfms] = size(imgs);
nfms = nfms - 1;
spx = zeros(rows, cols, nfms);
for i = 1:nfms
    spx(:, :, i) = vl_slic(single(imgs(:, :, :, i)), 30, 1);
end

M = zeros(rows, cols, 3, nfms);
for i = 1:nfms
    motion = sqrt(flow.hu(:, :, i) .* flow.hu(:, :, i) + ...
        flow.hv(:, :, i) .* flow.hv(:, :, i));
    t = kmeans2cls(motion);
    spx1 = spx(:, :, i);
    
    m = zeros(rows, cols);
    for k = 1:max(spx1(:))
        m(spx1(:) == k) = (mean(motion(spx1(:) == k)) > t);
    end
    M(:, :, :, i) = repmat(bwmorph(m, 'close', 10), [1, 1, 3]);
end

kt = 2;
imgs2 = zeros(rows, cols, 3, nfms - (K - 1), 'uint8');
imgs = double(imgs);
for i = K : nfms
    imgs1 = double(imgs(:, :, :, (i - K + 1) : (i - 1))) .* ...
        M(:, :, :, (i - K + 1) : (i - 1));
    W = sum(M(:, :, :, (i - K + 1) : (i - 1)), 4);
    F = repmat(bwmorph(W(:, :, 1) >= kt, 'close', 10), [1, 1, 3]);
    im1 = sum(imgs1, 4);
    im2 = imgs(:, :, :, i);
    im3 = (im2 .* (F > 0) + im1 .* (F > 0)) ./ (W + 1) + im2 .* (F < 1);
    imgs2(:, :, :, i - K + 1) = uint8(im3);
end
end
