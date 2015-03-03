init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
% frame_path_K = '/research/action_videos/video_data/ucf101_K5TR_frames';
frame_path_K = '/research/action_videos/video_data/ucf101_K10TR_frames';
% Writes out the video frames.
% out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);
idx = find(used_for_testing ~= 1);
parfor i = 1:length(idx)
    vid = idx(i);

%     if ~exist([out_path filesep num2str(vid)], 'file')
%         mkdir([out_path filesep num2str(vid)]);
%     end
 
    if ~exist([frame_path filesep num2str(vid) '_K' num2str(K) '_frames.mat']); 
        frames = load([frame_path filesep num2str(vid) '_frames.mat']);
        flow = load([flow_path filesep num2str(vid) '_flow.mat']);
        flow = decompress_flow(flow.flow_int, flow.flow_frac);
        M = compute_fgmask2(frames.imgs, flow.hu, flow.hv);
        imgs = clap_frames(frames.imgs, M, K);
        save_frames([frame_path_K filesep num2str(vid) ...
            '_K' num2str(K) '_frames.mat'], imgs); 
%     else
%        frames = load([frame_path filesep num2str(vid) ...
%            '_K' num2str(K) '_frames.mat']); 
%        imgs = frames.imgs;
   end
    
%     for j = 1:size(imgs, 4)
%        imname = [out_path filesep num2str(j + K - 1) '.png'];
%        imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
%     end
end
