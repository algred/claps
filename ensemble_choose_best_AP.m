init_ucf101;
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));
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
N = 30;
yy = zeros(N * length(val_idx), 1);
for i = 1:length(val_idx)
    id = val_idx(i);
    yy((i - 1) * N + 1 : i * N) = class_labels(id) * ones(N, 1);
    n1 = size(S10_train.S{id}, 2);
    if N > n1
        tt = floor(N / n1);
        frame_idx = repmat(1:n1, tt, 1);
        frame_idx = [frame_idx(:); randsample(n1, N - n1 * tt)];
    else
        frame_idx = randsample(n1, N);
    end
    SS1(:,(i - 1) * N + 1 : i * N) = S1_train.S{id}(:, frame_idx + 10);
    SS5(:,(i - 1) * N + 1 : i * N) = S5_train.S{id}(:, frame_idx + 5);
    SS10(:,(i - 1) * N + 1 : i * N) = S10_train.S{id}(:, frame_idx);
end
AP1 = zeros(101, 1);
AP5 = AP1;
AP10 = AP1;
for i = 1:101 
    yy1 = (yy == i) + (yy ~= i) * -1;
	
    [recall precision info] = vl_pr(yy1, SS1(i, :)');
    AP1(i) = info.ap_interp_11;
    
    [recall precision info] = vl_pr(yy1, SS5(i, :)');
    AP5(i) = info.ap_interp_11;
    
    [recall precision info] = vl_pr(yy1, SS10(i, :)');
    AP10(i) = info.ap_interp_11;
end
[acc best_model_id] = max([AP1 AP5 AP10], [], 2);

%% For model calibration: computes mean and std for each class in each model.
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

test_idx = find(used_for_testing == 1);
mean_s1_sel = mean_s1(best_model_id == 1);
std_s1_sel = std_s1(best_model_id == 1);
mean_s5_sel = mean_s5(best_model_id == 2);
std_s5_sel = std_s5(best_model_id == 2);
mean_s10_sel = mean_s10(best_model_id == 3);
std_s10_sel = std_s10(best_model_id == 3);

%% In testing, for each class, uses the model that works best on validation.
S1 = load([score_path filesep prefix 'aug_K' num2str(1) ...
    '_scores_' timestamp '.mat']);
S5 = load([score_path filesep prefix 'aug_K' num2str(5) ...
    '_scores_' timestamp '.mat']);
S10 = load([score_path filesep prefix 'aug_K' num2str(10) ...
    '_scores_' timestamp '.mat']);

C = zeros(101);
test_idx = find(used_for_testing == 1);
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



