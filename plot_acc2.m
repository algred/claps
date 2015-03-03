init_ucf101;
A1 = load('oxford_wframe_acc.mat');
A2 = load('data/oxford_augK1_acc.mat');
A = [A1.acc1 A2.acc];
for i = 1:4
    figure;
    if i == 4
        class_names1 = class_names((i-1) * 25 + 1 : 101);
        bar(A((i-1) * 25 + 1 : 101, :), 'grouped', 'Horizontal','on');
        ylim([0, 27]);
        set(gca, 'YTick', 1:26, 'yticklabel',class_names1);
    else
        class_names1 = class_names((i-1) * 25 + 1 : i * 25);
        bar(A((i-1) * 25 + 1 : i * 25, :), 'grouped', 'Horizontal','on');
        ylim([0, 26]);
        set(gca, 'YTick', 1:25, 'yticklabel',class_names1);
    end
    set(gca, 'FontSize', 12, 'FontWeight', 'bold');
    legend('ORG', 'AUG', 'Location','SouthEast');
end

