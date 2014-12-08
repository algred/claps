init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
K = 10;

% Writes out the video frames.
out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);
train_idx = find(used_for_testing ~= 1);
for vid = 8918 %1:length(train_idx)
    %vid = train_idx(i);
    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    flow = load([flow_path filesep num2str(vid) '_flow.mat']);
    flow = decompress_flow(flow.flow_int, flow.flow_frac);
    M = compute_fgmask2(frames.imgs, flow.hu, flow.hv);
    imgs = clap_frames(frames.imgs, M, K);
    for j = 1:size(imgs, 4)
        % imname = [out_path filesep num2str(vid) filesep num2str(j + K - 1) '.png'];
        % imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
        imshow(imgs(:, :, :, j));
        pause;
    end
end

