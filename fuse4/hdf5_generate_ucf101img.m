addpath('..');
init_ucf101;
ORG_IMG_DIM = 256;
IMG_DIM = 224;
IMG_DIM_DIF = ORG_IMG_DIM - IMG_DIM;
chunk_size = 99;
img_root = pathstring('X:\image_data\ucf101_imgs_final');
out_root = pathstring('/research/wvaction/data/image_data/hdf5_ucf101img');

%% Gets the list of images.
image_list = cell(101, 1);
N = 500;
L = zeros(N * 101, 1);
for c = 1:101
    list1 = dir([img_root filesep class_names{c} filesep '*.png']);
    idx = datasample(1:length(list1), N);
    offset = (c-1) * N;
    for j = 1:N
        image_list{offset + j} = [class_names{c} filesep list1(idx(j)).name];
    end
    L(offset + 1 : offset + N) = c;
end

%% Output data.
nepoch = 6;
n = length(image_list);
parfor i = 1:nepoch
    fid = fopen([out_root filesep 'epoch_' num2str(i) '.txt'], 'w');
    idx_p = randperm(n);
    data_chunk = zeros(IMG_DIM, IMG_DIM, 12, chunk_size);
    label_chunk = zeros(1, 1, 1, chunk_size);
    ind = 1;
    chunk_count = 1;
    for v = 1:n
        id = idx_p(v);
        im = double(imread([img_root filesep image_list{id}]));
        im = imresize(im, [ORG_IMG_DIM, ORG_IMG_DIM], 'bilinear');
        imgs = zeros(IMG_DIM, IMG_DIM, 12);
        flip_im = (rand(1) > 0.5);
        for k = 1:4
            off_x = randsample(IMG_DIM_DIF, 1);
            off_y = randsample(IMG_DIM_DIF, 1);
            im1 = im(off_y : off_y+IMG_DIM-1, off_x : off_x+IMG_DIM-1, :);
            im1 = transform_image(im1);
            if flip_im
                im1 = im1(end:-1:1, :, :);
            end
            imgs(:,:,(k-1)*3 + 1) = im1(:,:,1) - 103.939;
            imgs(:,:,(k-1)*3 + 2) = im1(:,:,2) - 116.779;
            imgs(:,:,(k-1)*3 + 3) = im1(:,:,3) - 123.68;
        end
        data_chunk(:,:,:,ind) = imgs;
        label_chunk(1,1,1,ind) = L(id) - 1;

        if ind >= chunk_size || v == n 
            h5filename = [out_root filesep num2str(i) ...
                '_' sprintf('%05d.h5',chunk_count)];

            h5create(h5filename, '/data', ...
                [IMG_DIM IMG_DIM 12 Inf], 'Datatype', 'single', ...
                'ChunkSize', [IMG_DIM IMG_DIM 3 chunk_size]); 
            h5write(h5filename, '/data', single(data_chunk), ...
                [1 1 1 1], size(data_chunk));
            
            h5create(h5filename, '/label', [1,1,1,Inf], 'Datatype', ...
                'single', 'ChunkSize', [1,1,1,chunk_size]);  
            h5write(h5filename, '/label', single(label_chunk), ...
                [1 1 1 1], size(label_chunk));
       
            fprintf('%05d batch processed\n',chunk_count)
            fprintf(fid,'%s\n', h5filename);

            data_chunk = zeros(IMG_DIM, IMG_DIM, 12, chunk_size);
            label_chunk = zeros(1, 1, 1, chunk_size);
            chunk_count = chunk_count + 1;
            ind = 1;
        else   
            ind = ind + 1;
        end
    end
    fclose(fid);
end

