A = [acc1 acc5 acc10 acc];
for i = 1:4
    figure;
    class_names1 = class_names((i-1) * 25 + 1 : i * 25);
    bar(A((i-1) * 25 + 1 : i * 25, :), 'grouped', 'Horizontal','on');
    ylim([0, 26]);
    set(gca, 'YTick', 1:25, 'yticklabel',class_names1);
    set(gca, 'FontSize', 12, 'FontWeight', 'bold');
    legend('K1', 'K5', 'K10', 'Combined', 'Location','SouthEast');
end

