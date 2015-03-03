% Lists static action images to be augmented to UCF101 frames. 
image_root = pathstring('X:\image_data\ucf101_imgs_aug');
load('ucf101_data.mat');
cls_map = containers.Map(class_names, 0:100);

cls_list = dir(image_root);
fid = fopen(['data' filesep 'ucf101img_aug.txt'], 'w'); 
for i = 3:length(cls_list)
    cls_name = cls_list(i).name;
    cid = cls_map(cls_name);
    img_list = dir([image_root filesep cls_name filesep '*.png']);
    for j = 1:length(img_list)
        img_name = [image_root filesep cls_name filesep img_list(j).name];
        fprintf(fid, '%s %d\n', img_name, cid);
    end
end
fclose(fid);
