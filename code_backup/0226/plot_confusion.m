init_ucf101;
A0 = load('data/oxford_wframe_acc.mat');
A1 = load('data/ucf101aug_K1_acc.mat');
A5 = load('data/ucf101aug_K5_acc.mat');
A10 = load('data/ucf101aug_K10_acc.mat');
A = [A10.acc A5.acc A1.acc A0.acc1]; 
for i = 1:4
    figure;
    if i == 4
        class_names1 = class_names((i-1) * 25 + 1 : 101);
        barh(A((i-1) * 25 + 1 : 101, :), 'grouped', 'Horizontal','on');
        ylim([0, 27]);
        set(gca, 'YTick', 1:26, 'yticklabel',class_names1);
    else
        class_names1 = class_names((i-1) * 25 + 1 : i * 25);
        barh(A((i-1) * 25 + 1 : i * 25, :), 'grouped', 'Horizontal','on');
        ylim([0, 26]);
        set(gca, 'YTick', 1:25, 'yticklabel',class_names1);
    end
    set(gca, 'FontSize', 10, 'FontWeight', 'bold');
    legend('CNN2+Fuse10', 'CNN2+Fuse5', ...
        'CNN2:CNN1+WebImg', 'CNN1:VFrame+ImageNet', ...
        'Location','northoutside', 'Orientation', 'Horizontal');
end

