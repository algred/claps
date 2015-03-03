init_ucf101;
score_path = '/research/action_videos/video_data/deepnet_ucf101';
prefix = 'oxford_';
timestamp = '0219';
load('train_val_split.mat');

%% For each class, finds which CNN model works best on validation data.
S1_train = load([score_path filesep prefix 'aug_K' num2str(1) ...
    '_scores_train_' timestamp '.mat']);
S5_train = load([score_path filesep prefix 'aug_K' num2str(5) ...
    '_scores_train_' timestamp '.mat']);
S10_train = load([score_path filesep prefix 'aug_K' num2str(10) ...
    '_scores_train_' timestamp '.mat']);

C1 = zeros(101);
C5 = zeros(101);
C10 = zeros(101);
for i = 1:length(val_idx)
    id = val_idx(i);
    
    s1 = S1_train.S{id};
    [~, a1] = max(s1);
    a1 = a1(:); 
    b1 = accumarray(a1, 1);
    [~, c] = max(b1);
    C1(class_labels(id), c) = C1(class_labels(id), c) + 1;

    s5 = S5_train.S{id};
    [~, a5] = max(s5);
    a5 = a5(:); 
    b5 = accumarray(a5, 1);
    [~, c] = max(b5);
    C5(class_labels(id), c) = C5(class_labels(id), c) + 1;

    s10 = S10_train.S{id};
    [~, a10] = max(s10);
    a10 = a10(:); 
    b10 = accumarray(a10, 1);
    [~, c] = max(b10);
    C10(class_labels(id), c) = C10(class_labels(id), c) + 1;
end

SS = cell2mat(S1_train.S(val_idx)');
SS = SS';
std_s1 = std(SS);
mean_s1 = mean(SS);
std_s1 = std_s1(:);
mean_s1 = mean_s1(:);

SS = cell2mat(S5_train.S(val_idx)');
SS = SS';
std_s5 = std(SS);
mean_s5 = mean(SS);
std_s5 = std_s5(:);
mean_s5 = mean_s5(:);

SS = cell2mat(S10_train.S(val_idx)');
SS = SS';
std_s10 = std(SS);
mean_s10 = mean(SS);
std_s10 = std_s10(:);
mean_s10 = mean_s10(:);

p1 = diag(C1);
acc1 = p1(:) ./ sum(C1, 2);

p5 = diag(C5);
acc5 = p5(:) ./ sum(C5, 2);

p10 = diag(C10);
acc10 = p10(:) ./ sum(C10, 2);

[acc best_model_id] = max([acc1 acc5 acc10], [], 2);

% In testing, for each class, uses the model that works best on validation.
S1 = load([score_path filesep prefix 'aug_K' num2str(1) ...
    '_scores_' timestamp '.mat']);
S5 = load([score_path filesep prefix 'aug_K' num2str(5) ...
    '_scores_' timestamp '.mat']);
S10 = load([score_path filesep prefix 'aug_K' num2str(10) ...
    '_scores_' timestamp '.mat']);

C = zeros(101);
test_idx = find(used_for_testing == 1);
mean_s1_sel = mean_s1(best_model_id == 1);
std_s1_sel = std_s1(best_model_id == 1);
mean_s5_sel = mean_s5(best_model_id == 2);
std_s5_sel = std_s5(best_model_id == 2);
mean_s10_sel = mean_s10(best_model_id == 3);
std_s10_sel = std_s10(best_model_id == 3);

for i = 1:length(test_idx)
    id = test_idx(i);
    s1 = S1.S{id}(:, 11:end);
    s5 = S5.S{id}(:, 6:end);
    s10 = S10.S{id};
    n = size(S10.S{id}, 2);

    s = zeros(size(s1));
    s(best_model_id == 1, :) = (s1(best_model_id == 1, :) ...
        - repmat(mean_s1_sel, 1, n)) ./ repmat(std_s1_sel, 1, n);
    s(best_model_id == 2, :) = (s5(best_model_id == 2, :) ...
        - repmat(mean_s5_sel, 1, n)) ./ repmat(std_s5_sel, 1, n);
    s(best_model_id == 3, :) = (s10(best_model_id == 3, :) ...
        - repmat(mean_s10_sel, 1, n)) ./ repmat(std_s10_sel, 1, n);

    [~, a] = max(s);
    b = accumarray(a(:), 1);
    [~, c] = max(b);
    C(class_labels(id), c) = C(class_labels(id), c) + 1;
end

p = diag(C);
acc = p(:) ./ sum(C, 2);



