init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
vid = 2300;
load([frame_path filesep num2str(vid) '_frames.mat']);
load([flow_path filesep num2str(vid) '_flow.mat']);
flow = decompress_flow(flow_int, flow_frac);
[rows, cols, chn, nfms] = size(imgs);
nfms = nfms - 1;

M = zeros(rows, cols, 3, nfms);
for i = 1:nfms
    m = sqrt(flow.hu(:, :, i) .* flow.hu(:, :, i) + ...
        flow.hv(:, :, i) .* flow.hv(:, :, i));
    t = kmeans2cls(m);
    m = (m - t);
    M(:, :, :, i) = repmat(m > t, [1, 1, 3]);
end

%%
M = zeros(rows, cols, nfms);
for i = 1:nfms
    motion = sqrt(flow.hu(:, :, i) .* flow.hu(:, :, i) + ...
        flow.hv(:, :, i) .* flow.hv(:, :, i));
    t = kmeans2cls(motion);
    spx1 = spx(:, :, i);
    
    m = zeros(rows, cols);
    for k = 1:max(spx1(:))
        m(spx1(:) == k) = (mean(motion(spx1(:) == k)) > t);
    end
    M(:, :, i) = bwmorph(m, 'close', 10);
end

%%
K = 5;
% kt = round(K / 2);
kt = 1;
for i = K : nfms
    Ws = false(rows, cols, K);
    Ws(:, :, K) = M(:, :, i) > 0;
    for j = 1 : K - 1
        Ws(:, :, K - j) = (M(:, :, i - j) > 0) & (~any(Ws(:, :, K - j + 1 : K), 3)); 
    end
    W = sum(Ws, 3);
    
    im1 = zeros(rows, cols, 3);
    for j = 1:K
        im1 = im1 + im1 .* repmat(Ws(:, :, j), [1 1 3]);
    end
    im3 = im1 + double(imgs(:, :, :, i)) .* repmat((W < 1), [1 1 3]);
    imshow(uint8(im3), 'InitialMagnification', 100);
    pause;
end