function m = compute_fgmask(im, hu, hv)
spx = vl_slic(single(im), 30, 1);
motion = sqrt(hu .* hu + hv .* hv);
t = kmeans2cls(motion);
spx = spx(:);
m = false(rows, cols);
m(spx == k) = (mean(motion(spx == k)) > t);
m = bwmorph(m, 'close', 10);
end

