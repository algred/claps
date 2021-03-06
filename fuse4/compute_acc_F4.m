addpath('..');
init_ucf101;
score_path = pathstring('/research/action_videos/video_data/deepnet_ucf101');

load([score_path filesep 'ucf101augVGG16Fuse4V3Split2_iter110000_scores']);
% load([score_path filesep 'ucf101_AUG_VGG16_F4_iter70000_scores_0312.mat']);
C = zeros(101);
for i = 1:length(video_list)
    if used_for_testing(i) ~= 2
        continue;
    end
%     [~, a] = max(S{i});
%     a = a(:); 
%     b = accumarray(a, 1);
%     [~, c] = max(b);
%     C(class_labels(i), c) = C(class_labels(i), c) + 1;
%  
%   [~, pred] = max(mean(S{i}, 2), [], 1);
   [~, pred] = max(S{i}, [], 1);
   C(class_labels(i), pred) = C(class_labels(i), pred) + 1;
%    [~, pred] = max(S{i}(:, NID), [], 1);
%    C(class_labels(i), pred) = C(class_labels(i), pred) + 1;

end
acc = diag(C) ./ sum(C, 2);

