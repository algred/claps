function IX = rand_idx(intv, K, N, max_val)
% Samples N index arrays: IX is N * K.
intv2 = floor(intv / 2);
IX = randn(N, K) * sqrt(intv2 - 1) + ...
    repmat(intv2 : intv : intv2 + (K-1) * intv, N, 1);

MIN = repmat(1:intv:1+(K-1)*intv, N, 1);
MAX = repmat(intv:intv:K*intv, N, 1);
MAX(:, end) = min(MAX(:, end), max_val);
IX = min(MAX, max(round(IX), MIN));

