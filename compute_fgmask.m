function m = compute_fgmask(im, hu, hv)
spx = vl_slic(single(im), 30, 1);
motion = sqrt(hu .* hu + hv .* hv);
t = kmeans2cls(motion);
spx = spx(:);
m = false(size(hu));
for k = 1:max(spx(:))
    m(spx == k) = (mean(motion(spx == k)) > t);
end
m = bwmorph(m, 'close', 10);
end

