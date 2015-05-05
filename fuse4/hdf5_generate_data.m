% Change history: 
% 03/15/15 -  1) Added flipping and random cropping.
%             2) Generate data into 3 splits. 

addpath('..');
init_ucf101;
ORG_IMG_DIM = 256;
IMG_DIM = 224;
IMG_DIM_DIF = ORG_IMG_DIM - IMG_DIM;
chunk_size = 99;
out_root = pathstring(['X:\video_data\deepnet_ucf101\fuse4_hdf5\split'...
    num2str(split_id)]);

%% Output data.
idx = find(used_for_testing == split_id);
n = length(idx);
nepoch = 24;
parfor i = 1:nepoch
    fid = fopen([out_root filesep 'epoch_' num2str(i) '.txt'], 'w');
    idx_p = randperm(n);
    data_chunk = zeros(IMG_DIM, IMG_DIM, 12, chunk_size);
    label_chunk = zeros(1, 1, 1, chunk_size);
    ind = 1;
    chunk_count = 1;
    for v = 1:n
        vid = idx(idx_p(v));
        frames = load([frame_path filesep num2str(vid) '_frames.mat']);
        nfms = size(frames.imgs, 4);
        intv = floor(nfms / 4);
        IX = rand_idx(intv, 4, 1, nfms);
        imgs = zeros(IMG_DIM, IMG_DIM, 12);
        off_x = randsample(IMG_DIM_DIF, 1);
        off_y = randsample(IMG_DIM_DIF, 1);
        flip_im = (rand(1) > 0.5);
        for k = 1:4
            im = double(frames.imgs(:,:,:,IX(1, k)));
            im = imresize(im, [ORG_IMG_DIM, ORG_IMG_DIM], 'bilinear');
            im = im(off_y : off_y+IMG_DIM-1, off_x : off_x+IMG_DIM-1, :);
            im = transform_image(im);
            if flip_im
                im = im(end:-1:1, :, :);
            end
            imgs(:,:,(k-1)*3 + 1) = im(:,:,1) - 103.939;
            imgs(:,:,(k-1)*3 + 2) = im(:,:,2) - 116.779;
            imgs(:,:,(k-1)*3 + 3) = im(:,:,3) - 123.68;
        end
        data_chunk(:,:,:,ind) = imgs;
        label_chunk(1,1,1,ind) = class_labels(vid) - 1;

        if ind >= chunk_size || v == n 
            h5filename = [out_root filesep num2str(i) ...
                '_' sprintf('%05d.h5',chunk_count)];

            h5create(h5filename, '/data', ...
                [IMG_DIM IMG_DIM 12 Inf], 'Datatype', 'single', ...
                'ChunkSize', [IMG_DIM IMG_DIM 12 chunk_size]); 
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

