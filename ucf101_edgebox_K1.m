% Merges edgeboxs of K consecutive frames that significantly overlap.
init_ucf101;
K = 1;
overlap_thre = 0.2;

edgebox_path = pathstring('/research/action_data/ucf101-edgebox');
bboxs = cell(length(video_list), 1);
for vid = 8300:length(video_list)
    frames = load([frame_path filesep num2str(vid) '_frames.mat']);
    nfms = size(frames.imgs, 4) - 1;
    all_bboxs = load([edgebox_path filesep num2str(vid) '_bboxs.mat']);
    all_bboxs = all_bboxs.bboxs(1:nfms);
    all_bboxs = all_bboxs(:);
    merged_bboxs = cell(nfms - K + 1, 1); 
    for j = K : nfms
        this_bboxs = cell2mat(all_bboxs(j - K + 1 : j));
        bboxs1 = this_bboxs(:, 1:4);
        while true
            % Computes overlaps as:
            % intersection / min(area_of_box1, area_of_box2)
            n = size(bboxs1, 1);
            IA = rectint(bboxs1(:, 1:4), bboxs1(:, 1:4));
            A = bboxs1(:, 3) .* bboxs1(:, 4);
            MA = min(repmat(A, 1, n), repmat(A', n, 1));
            F = IA ./ MA;
            % UA = repmat(A, 1, n) + repmat(A', n, 1) - IA;
            % F = IA ./ UA;
            
            % Merges bounding boxes that significantly overlap.
            [S, C] = graphconncomp(sparse(F >= overlap_thre), 'Directed', false);
            if S == n
                break;
            end
            cidx = unique(C);
            new_bboxs = zeros(S, 4);
            for cid = 1:length(cidx)
                B = bboxs1(C == cidx(cid), 1:4);
                x1 = min(B(:, 1));
                y1 = min(B(:, 2));
                x2 = max(B(:, 1) + B(:, 3) - 1);
                y2 = max(B(:, 2) + B(:, 4) - 1);
                new_bboxs(cid, :) = [x1 y1 x2 - x1 + 1 y2 - y1 + 1];
            end
            bboxs1 = new_bboxs;
        end
        
        new_bboxs = bboxs1;
        merged_bboxs{j - K + 1} = new_bboxs;
        imshow(frames.imgs(:, :, :, j));
        hold on; 
        for i = 1:size(new_bboxs, 1)
            rectangle('Position', new_bboxs(i, :), 'EdgeColor', 'y', 'LineWidth', 1.5); 
        end
        for i = 1:size(this_bboxs, 1)
            rectangle('Position', this_bboxs(i, 1:4), 'EdgeColor', 'r', 'LineWidth', 1); 
        end
        hold off;
        pause(0.2);
    end
    bboxs{vid} = merged_bboxs;
end
save(pathstring([edgebox_path filesep 'merged_bboxs_K' ...
    num2str(K) '.mat']), 'bboxs');

