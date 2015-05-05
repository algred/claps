% Augment the UCF101_IMG dataset by cropping. 
image_root = pathstring('X:\image_data\ucf101_imgs_final');
out_root = pathstring('X:\image_data\ucf101_imgs_aug2');
load('ucf101_data.mat');
IMAGE_DIM = 256;
DELTA = 30;
N = 6;

parfor i = 1:length(class_names)
    cls_name = class_names{i};
    out_dir = [out_root filesep cls_name];
    if ~exist(out_dir, 'file')
        mkdir(out_dir);
    end
    img_list = dir([image_root filesep cls_name]);
    
    for j = 1:length(img_list)
        img_name = img_list(j).name;
        if strcmp(img_name, '.') || strcmp(img_name, '..')
            continue;
        end
        ind = 1;
        [~,fname, extstr] = fileparts(img_name);
        try
            im = imread([image_root filesep cls_name filesep img_name]);
            rows = size(im, 1);
            cols = size(im, 2);
            if rows > cols
                im = imresize(im, [round(rows * (IMAGE_DIM / cols)), IMAGE_DIM]);
            else
                im = imresize(im, [IMAGE_DIM, round(cols * (IMAGE_DIM / rows))]);
            end
            rows = size(im, 1);
            cols = size(im, 2);
            delta = max(rows, cols) - IMAGE_DIM;
            if  delta > DELTA
                intv = floor(delta / N);
                imgs = cell(N, 1);
                if rows > cols
                    for k = 1:N
                        imgs{k} = imcrop(im, [1, intv * k + 1, IMAGE_DIM, IMAGE_DIM]);
                    end
                else
                    for k = 1:N
                        imgs{k} = imcrop(im, [intv * k + 1, 1, IMAGE_DIM, IMAGE_DIM]);
                    end
                end
                for k = 1:N
                    imgs{k} = imresize(imgs{k}, [IMAGE_DIM, IMAGE_DIM]);
                    imwrite(imgs{k}, [out_dir filesep fname '_' num2str(ind) '.png'], 'png');
                    ind = ind + 1;
                end
            end
            im = imresize(im, [IMAGE_DIM, IMAGE_DIM]);
            imwrite(im, [out_dir filesep fname '_' num2str(ind) '.png'], 'png');
            ind = ind + 1;
        catch exception
            getReport(exception)
            fprintf('ERROR: %s - %s\n', cls_name, img_name);
        end
    end
end

