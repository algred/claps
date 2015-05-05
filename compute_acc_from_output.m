init_ucf101;
score_path = pathstring('/research/action_videos/video_data/deepnet_ucf101');
output_path = pathstring('X:\video_data\deepnet_ucf101\cnnoutput\split1\medium_run1');
C1 = zeros(101);
split = 1;
for i = 1:length(video_list)
    if used_for_testing(i) ~= split
        continue;
    end
    try
    load([output_path filesep num2str(i) '_out.mat']);   
    catch exception
        getReport(exception)
        continue;
    end
    [~, a1] = max(S);
    a1 = a1(:); 
    b1 = accumarray(a1, 1);    
    [~, c] = max(b1);
    C1(class_labels(i), c) = C1(class_labels(i), c) + 1;
end

p = diag(C1);
acc = (p(:) ./ sum(C1, 2));
