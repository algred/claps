init_ucf101;
addpath(genpath(pathstring('Y:\tools\vlfeat-0.9.16-bin\vlfeat-0.9.16\toolbox')));
frame_path = '/research/action_data/ucf101-frames'
K = 10;
out_path = pathstring(['/research/action_data/ucf101-k' num2str(K)]);

% Writes out the video frames.
idx = find(used_for_testing == 1);
parfor i = 1:length(idx)
    vid = idx(i);

    if ~exist([out_path filesep num2str(vid)], 'file')
        mkdir([out_path filesep num2str(vid)]);
    end

    frames = load([frame_path filesep num2str(vid) ...
        '_K' num2str(K) '_frames.mat']); 
    imgs = frames.imgs;
  
    for j = 1:size(imgs, 4)
       imname = [out_path filesep num2str(vid) ...
           filesep num2str(j + K - 1) '.png'];
       imwrite(imresize(imgs(:, :, :, j), [256, 256]), imname, 'png');
    end
end
