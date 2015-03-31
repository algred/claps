% Augment the UCF101_IMG dataset by cropping.
init_ucf101;
image_root = pathstring('X:\image_data\ucf101_imgs');
out_root = pathstring('X:\image_data\ucf101_imgs_final');

parfor i = 1:length(class_names)
    img_list = dir([image_root filesep class_names{i}]);
    out_dir = [out_root filesep class_names{i}];
    if ~exist(out_dir, 'file')
        mkdir(out_dir);
    end
    ind = 1;
    for j = 3:length(img_list)
        try
            im = cv.imread([image_root filesep class_names{i} filesep ...
                img_list(j).name]);
            if ndims(im) < 3 || max(im(:)) > 255
                fprintf('READ ERROR: %s\n', [image_root filesep ...
                    class_names{i} filesep img_list(j).name]);
                continue;
            end
            imwrite(im, [out_dir filesep class_names{i} ...
                '_' num2str(ind) '.png'], 'png');
            ind = ind + 1;
        catch exception
            fprintf('ERROR: %s\n', [image_root filesep ...
                class_names{i} filesep img_list(j).name]);
            continue;
        end
    end
end


