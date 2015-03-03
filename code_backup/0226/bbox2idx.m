function idx = bbox2idx(bbox, rows, cols)
bbox(3:4) = bbox(1:2) + bbox(3:4) - 1;
idx = sub2ind([rows, cols], [bbox(2) : bbox(4), bbox(1) : bbox(3)];
idx = idx(:);
end
