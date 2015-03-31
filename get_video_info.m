init_ucf101;
parfor vid = 1:length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    [h, w, ~, nfms] = size(frames.imgs);
    video_info(vid).w = w;
    video_info(vid).h = h;
    video_info(vid).nfms = nfms;
end
save('video_info.mat', 'video_info');
