addpath('hmdb');
init_hmdb;

idf_extractor = '/research/wvaction/tools/improved_trajectory_release/release/DenseTrackStab';
out_root = '/research/action_features/hmdb/idf';

parfor i = 1:12
setenv('LD_LIBRARY_PATH', '.:/usr/local/lib:/usr/local/cuda-6.5/lib64:/home/grad2/shugaoma/local/lib:/home/grad2/shugaoma/local/opencv/lib:/home/grad2/shugaoma/local/gflags/lib:/home/grad2/shugaoma/local/ffmpeg/lib')
end

parfor vid = id1:id2
    out_name = [out_root filesep num2str(vid) '_idf'];
    if exist(out_name, 'file')
        continue;
    end
    system([idf_extractor ' ' video_list(vid).video_name ' > ' out_name]);
end
