init_ucf101;
score_path = '/research/action_videos/video_data/deepnet_ucf101';

load([score_path filesep 'verydeep_augVGG16S4All_iter40000_scores_0311']);
C = zeros(101);

for i = 1:length(video_list)
    if used_for_testing(i) ~= 1
        continue;
    end

    [~, pred] = max(mean(S{i}, 2), [], 1);
    C(class_labels(i), pred) = C(class_labels(i), pred) + 1;
end

acc = diag(C) ./ sum(C, 2);

