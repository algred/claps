init_ucf101;
score_path = '/research/action_videos/video_data/deepnet_ucf101';
prefix = 'oxford_';
timestamp = '_0219';

S1 = load([score_path filesep prefix 'aug_K' num2str(1) ...
    '_scores' timestamp '.mat']);

fprintf('%s\n', prefix);
C1 = zeros(101);

for i = 1:length(video_list)
    if used_for_testing(i) ~= 1
        continue;
    end
    s1 = S1.S{i};
   
    [~, a1] = max(s1);
    a1 = a1(:); 
   
    b1 = accumarray(a1, 1);
    
    [~, c] = max(b1);
    C1(class_labels(i), c) = C1(class_labels(i), c) + 1;
end

p = diag(C1);
acc = (p(:) ./ sum(C1, 2));

