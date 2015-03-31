class_names = {'applauding', 'blowing_bubbles', ...
    'cooking', 'cutting_trees', ...
    'drinking', 'feeding_a_horse', 'fishing', 'fixing_a_bike', 'gardening', ...
    'holding_an_umbrella', 'jumping', 'looking_through_a_microscope', ...
    'phoning', 'pouring_liquid', 'pushing_a_cart', 'reading', ...
    'running', 'smoking', 'taking_photos', 'texting_message', ...
    'throwing_frisby', 'using_a_computer', 'washing_dishes', ...
    'watching_TV', 'waving_hands'};

img_root = pathstring('/research/wvaction/data/image_data/Standford40/JPEGImages'); 
out_root = pathstring('/research/wvaction/data/image_data/Standford40/normalized_imgs');
test_fid = fopen(['data' filesep 'standford40_test.txt'], 'w');
train_fid = fopen(['data' filesep 'standford40_train.txt'], 'w');

IMG_DIM = 256;
K = 20;
for c = 1:length(class_names)
    img_list = dir([img_root filesep class_names{c} '*.jpg']);
    n = length(img_list);
    fprintf('%d\n', n);
    test_idx = randsample(n, K);
    train_idx = setdiff(1:n, test_idx);

    for i = 1:n
        im = imread([img_root filesep img_list(i).name]);
        im = imresize(im, [IMG_DIM, IMG_DIM]);
        imwrite(im, [out_root filesep img_list(i).name], 'jpg');
    end

    for i = 1:length(test_idx)
        fprintf(test_fid, '%s %d\n', ...
            [out_root filesep img_list(test_idx(i)).name], c + 100);
    end

    for i = 1:length(train_idx)
        fprintf(train_fid, '%s %d\n', ...
            [out_root filesep img_list(train_idx(i)).name], c + 100);
    end
end
fclose(train_fid);
fclose(test_fid);
