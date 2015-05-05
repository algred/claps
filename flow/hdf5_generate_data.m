addpath('..');
addpath('../external');
init_ucf101;

IMG_DIM = 256;
out_root = pathstring('/scratch/shugao/ucf101-flow-hdf5');
flow_path = '/research/action_features/ucf101-flow'; 

%% Output data.
N = 30;
K = 5;
parfor vid = 1:length(video_list)
    out_dir = [out_root filesep num2str(vid)];
    if ~exist(out_dir, 'file')
        mkdir(out_dir);
    end
    flow = load([flow_path filesep num2str(vid) '_flow.mat']);
    flow = decompress_flow(flow.flow_int, flow.flow_frac);
    nfms = size(flow.u, 3);
    intv = round(nfms / N);
    for i = 1 : intv : (nfms - K)
        data_chunk = zeros(IMG_DIM, IMG_DIM, 2 * K, 1, 'single');
        label_chunk = zeros(1, 1, 1, 1, 'single');
        for j = 1:K
            u = imresize(single(flow.u(:, :, i + j - 1)), ...
                [IMG_DIM, IMG_DIM], 'bilinear');
            v = imresize(single(flow.v(:, :, i + j - 1)), ...
                [IMG_DIM, IMG_DIM], 'bilinear');
            data_chunk(:, :, j * 2 - 1, 1) = u - mean(u(:));
            data_chunk(:, :, j * 2, 1) = v - mean(v(:));
        end
        label_chunk(1, 1, 1, 1) = class_labels(vid) - 1;

        h5filename = [out_dir filesep sprintf('%03d.h5', i)];

        h5create(h5filename, '/data', [IMG_DIM IMG_DIM 2*K inf], ...
            'Datatype', 'single', 'ChunkSize', [IMG_DIM IMG_DIM 2*K 1]); 
        h5write(h5filename, '/data', single(data_chunk), ...
            [1 1 1 1], [IMG_DIM IMG_DIM 2*K, 1]);
        
        h5create(h5filename, '/label', [1,1,1,inf], 'Datatype', 'single', ...
            'ChunkSize', [1 1 1 1]);  
        h5write(h5filename, '/label', single(label_chunk), ...
            [1 1 1 1], [1 1 1 1]);
    end
end

