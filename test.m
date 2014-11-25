init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
vid = 2300;
load([frame_path filesep num2str(vid) '_frames.mat']);
load([flow_path filesep num2str(vid) '_flow.mat']);
flow = decompress_flow(flow_int, flow_frac);
[rows, cols, chn, nfms] = size(imgs);
nfms = nfms - 1;

spx = zeros(rows, cols, nfms);
for i = 1:nfms
    spx(:, :, i) = vl_slic(single(imgs(:, :, :, i)), 30, 1);
%     perim = true(rows, cols);
%     for k = 1:max(spx(:))
%         perimK = bwperim(spx == k);
%         perim(perimK) = false;
%     end
%     im1 = imgs(:, :, :, i) .* uint8(cat(3, perim, perim, perim));
%     imshow(im1);
%     pause;
end

% M = zeros(rows, cols, 3, nfms);
% for i = 1:nfms
%     m = sqrt(flow.hu(:, :, i) .* flow.hu(:, :, i) + ...
%         flow.hv(:, :, i) .* flow.hv(:, :, i));
%     t = kmeans2cls(m);
%     M(:, :, :, i) = repmat(m > t, [1, 1, 3]);
% end

%%
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

%%
K = 5;
% kt = round(K / 2);
kt = 1;
for i = K : nfms
    imgs1 = double(imgs(:, :, :, (i - K + 1) : (i - 1))) .* ...
        M(:, :, :, (i - K + 1) : (i - 1));
    W = sum(M(:, :, :, (i - K + 1) : (i - 1)), 4);
    F = repmat(bwmorph(W(:, :, 1) >= kt, 'close', 10), [1, 1, 3]);
    im1 = sum(imgs1, 4);
    im2 = double(imgs(:, :, :, i));
    im3 = (im2 .* (F > 0) + im1 .* (F > 0)) ./ (W + 1) + im2 .* (F < 1);
    imshow(uint8(im3), 'InitialMagnification', 100);
    pause;
end