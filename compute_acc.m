init_ucf101;
score_path = '/research/action_videos/video_data/deepnet_ucf101';
% prefix = 'oxford_';
% timestamp = '_0219';
% K = 1;
% S1 = load([score_path filesep prefix 'aug_K' num2str(K) ...
%     '_scores' timestamp '.mat']);
% fprintf('%s\n', prefix);

S1 = load([score_path filesep 'verydeep_augK1all_iter80000_scores_0226.mat']); 
% S1 = load([score_path filesep 'oxford_augK1all_scores_0226.mat']); 
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
% imagesc(C1);
% set(gca, 'FontSize', 6);
% set(gca, 'YTick', 1:101, 'yticklabel',class_names);
% set(gca, 'XTick', 1:101, 'xticklabel',class_names);
% h = gca;
% a=get(h,'XTickLabel');
% set(h,'XTickLabel',[]);
% b=get(h,'XTick');
% c=get(h,'YTick');
% th=text(b,repmat(c(1)-.1*(c(2)-c(1)),length(b),1) + 101, ...
%     a, 'HorizontalAlignment','right','rotation',60, ...
%     'FontSize', 6);
