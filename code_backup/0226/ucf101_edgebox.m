% For each frame, compute at most N bbox proposals that significantly overlap
% with one connected component of the foreground mask.

init_ucf101;
addpath(genpath('/research/wvaction/code/toolbox'));
addpath('/home/grad2/shugaoma/lib/edgebox/release');
addpath(genpath('/research/wvaction/code/vlfeat/vlfeat-0.9.18/'));

opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e3;  % max number of boxes to detect
model=load('/home/grad2/shugaoma/lib/edgebox/release/models/forest/modelBsds');
model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=1;

IOU_thre = 0.5;
out_path = pathstring(['/research/action_data/ucf101-edgebox']);
for vid = id1:id2 %1:length(video_list)
    if exist([out_path filesep num2str(vid) '_bboxs.mat'], 'file')
        continue;
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    flow = load([flow_path filesep num2str(vid) '_flow.mat']);
    flow = decompress_flow(flow.flow_int, flow.flow_frac);
    rows = size(frames.imgs, 1);
    cols = size(frames.imgs, 2);
    nfms = size(frames.imgs, 4) - 1;
    bboxs = cell(nfms, 1);
    parfor j = 1:nfms
        % Computes foreground mask.
        mask = compute_fgmask(frames.imgs(:, :, :, j), ...
            flow.hu(:, :, j), flow.hv(:, :, j));
        CC = bwconncomp(mask);
        cc_bboxs = zeros(CC.NumObjects, 4);
        for k = 1:CC.NumObjects
            [y, x] = ind2sub([rows, cols], CC.PixelIdxList{k});
            x1 = min(x); x2 = max(x);
            y1 = min(y); y2 = max(y);
            cc_bboxs(k, :) = [x1 y1 x2 - x1 + 1  y2 - y1 + 1];
        end

        % Computes bbox proposals.  
        this_bboxs = edgeBoxes(frames.imgs(:, :, :, j), model, opts);

        % Selects bboxes.
        IA = rectint(this_bboxs(:, 1:4), cc_bboxs);
        AA = -repmat(this_bboxs(:, 3) .* this_bboxs(:, 4), 1, CC.NumObjects)...
            + repmat((cc_bboxs(:, 3) .* cc_bboxs(:, 4))', size(this_bboxs, 1), 1);
        IOU = max(IA ./ (AA - IA), [], 2);
        IOU = IOU(:);
        bboxs{j} = [this_bboxs(IOU >= IOU_thre, :) IOU(IOU >= IOU_thre)];
    end
    save([out_path filesep num2str(vid) '_bboxs.mat'], 'bboxs');
end

