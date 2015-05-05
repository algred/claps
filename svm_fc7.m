function svm_fc7(cid, split, mode)
load('ucf101_data.mat');
addpath('/research/wvaction/tools/liblinear-1.92/matlab/');
feat_path = ['/research/action_videos/video_data/deepnet_ucf101/cnnfeat/split'...
    num2str(split) '/fc6'];
out_path = ['/research/action_videos/video_data/deepnet_ucf101/cnnfeat/split'...
    num2str(split)];

% Loads the encoded data.
data_file = [out_path filesep 'fc7_max_avg.mat']; 
load(data_file);

% Prepare the data.
fprintf('Prepare the data...\n');
if strcmp(mode, 'avg_pool')
    D = A;
elseif strcmp(mode, 'max_pool')
    D = M;
else
    fprintf('Mode not recognized.\n');
    keyboard;
end
N = 20;
DIM = size(D{1}, 2);
X = zeros(length(D), DIM * N);
for i = 1:length(D)
    n = size(D{i}, 1);
    if n > N
        idx = randsample(n, N);
    elseif n < N
        idx = sort(datasample(1:n, N));
    else
        idx = 1:N;
    end
    X1 = D{i}(idx, :);
    X1 = X1';
    X(i, :) = X1(:)';
end

% SVM training.
fprintf('Start the training...\n');
C = 2.^[-5:5];
DIM = size(X, 2);
ncls = 101;
ACC = zeros(length(C), ncls);

train_flg = (used_for_testing ~= split);
test_labels = class_labels(~train_flg); test_labels = test_labels(:);
train_labels = class_labels(train_flg); train_labels = train_labels(:);

test_counts = zeros(ncls);
for c = 1:ncls
    test_counts(c) = sum(test_labels == c);
end
test_total = sum(test_counts);

svm_c = C(cid);
wc = zeros(DIM+1, ncls);
acc = zeros(ncls, 1);
for c = 1:ncls
    fprintf('Train SVM for class %d...', c);
    y = (train_labels == c) + (train_labels ~= c) * -1;
    kp = 1/sum(y>0); kn = 1/sum(y<0);
    options = ['-s 3 -q -B 1 -c ' num2str(svm_c)...
        ' -w1 ' num2str(kp) ' -w-1 ' num2str(kn)];
    model = train(y, sparse(X(train_flg, :)), options);
    wc(:, c) = model.w(:) * model.Label(1);
    fprintf('Done\n');
end
P = [X(~train_flg, :) ones(test_total, 1)] * wc;
[V, I] = max(P, [], 2);
for c = 1:ncls
    acc(c) = sum(I == c & test_labels == c) / test_counts(c);
end
fprintf('C = %f, mACC = %f\n', svm_c, mean(acc));
save([out_path filesep 'svm_fc7_models_C_' num2str(svm_c) ...
    '_split' num2str(split)], 'wc', 'acc');
