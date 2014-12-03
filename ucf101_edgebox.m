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

N = 100;
IOU_thre = 0.5;
out_path = pathstring(['/research/action_data/ucf101-edgebox']);
for vid = 1:1 %length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    flow = load([flow_path filesep num2str(vid) '_flow.mat']);
    flow = decompress_flow(flow.flow_int, flow.flow_frac);
    rows = size(frames.imgs, 1);
    cols = size(frames.imgs, 2);
    nfms = size(frames.imgs, 4);
    bboxes = cell(nfms, 1);
    for j = 1:nfms
        % Computes foreground mask.
        mask = compute_fgmask(frames.imgs(:, :, :, j), flow.hu, flow.hv);
        CC = bwconncomp(mask);
        cc_masks = false(rows * cols, CC.NumObjects);
        cc_sz = zeros(1, CC.NumObjects)
        for k = 1:CC.NumObjects
            cc_masks(CC.PixelIdxList{k}, k) = true;
            cc_sz(k) = length(CC.PixelIdxList{k});
        end

        % Computes bbox proposals.  
        this_bboxs = edgeBoxes(im, model, opts);

        % Selects bboxes.
        selected_boxes = [];
        num_selected = 0;
        for k = 1:size(this_bboxs, 1)
            idx = bbox2idx(this_bboxs(k, 1:4), rows, cols));
            IX = sum(cc_masks(idx, :), 2);
            UX = length(idx) + cc_sz - IX;
            IOU = IX ./ UX;
            if any(IOU >= IOU_thre)
                selected_boxes = [selected_boxes; this_bboxs(k, 1:4)];
                num_selected = num_selected + 1;
                if num_selected >= N
                    break;
                end
            end
        end
        bboxs{j} = selected_boxes;
    end
    save([out_path filesep num2str(vid) '_bboxs.mat'], 'bboxs');
end

