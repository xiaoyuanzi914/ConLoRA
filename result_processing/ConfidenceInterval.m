function [mean_accuracies, confidence_intervals] = calculate_confidence_intervals(accuracy_data, confidence_level)
    if nargin < 2
        confidence_level = 0.95;  % 默认置信水平
    end
    
    [epochs, clients] = size(accuracy_data); % epochs代表行，clients代表列
    mean_accuracies = zeros(1, clients); % 每一列的平均值
    confidence_intervals = zeros(clients, 2); % 每列的置信区间
    
    for client = 1:clients
        accuracies = accuracy_data(:, client);  % 取出每一列的数据
        mean_accuracy = mean(accuracies);       % 计算每一列的平均值
        standard_error = std(accuracies) / sqrt(epochs);  % 计算标准误差
        t_value = tinv(1 - (1 - confidence_level) / 2, epochs - 1);  % 查找t分布临界值
        
        % 计算置信区间
        margin_of_error = t_value * standard_error;
        confidence_interval = [mean_accuracy - margin_of_error, mean_accuracy + margin_of_error];
        
        mean_accuracies(client) = mean_accuracy;
        confidence_intervals(client, :) = confidence_interval;
    end
end

%此处需要输入自己的结果数据
acc_data_B1=[];
acc_data_B2=[];
acc_data_B3=[];

acc_data_BA1=[];
acc_data_BA2=[];
acc_data_BA3=[];
[mean_accuracies1, confidence_intervals1] = calculate_confidence_intervals(acc_data_B2);
[mean_accuracies2, confidence_intervals2] = calculate_confidence_intervals(acc_data_BA2);

function plot_accuracies_with_confidence_intervals(mean_accuracies_1, confidence_intervals_1,mean_accuracies_2, confidence_intervals_2)
    epochs = 1:length(mean_accuracies_1);
    
    % 提取置信区间的上下界
    lower_bounds_1 = confidence_intervals_1(:, 1);
    upper_bounds_1 = confidence_intervals_1(:, 2);
    
    lower_bounds_2 = confidence_intervals_2(:, 1);
    upper_bounds_2 = confidence_intervals_2(:, 2);

    figure('Position', [100, 100, 750, 500]); % 设置图的大小
    
    % 绘制 DLora 的均值和置信区间
    plot(epochs, mean_accuracies_1, 'b', 'DisplayName', 'ConLoRA', 'LineWidth', 1.5); 
    hold on;
    fill([epochs, fliplr(epochs)], [upper_bounds_1', fliplr(lower_bounds_1')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none','DisplayName','95% CI ConLoRA');
    
    % 绘制 Lora 的均值和置信区间
    plot(epochs, mean_accuracies_2, 'r', 'DisplayName', 'LoRA-DFL', 'LineWidth', 1.5); 
    fill([epochs, fliplr(epochs)], [upper_bounds_2', fliplr(lower_bounds_2')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none','DisplayName','95% CI LoRA-DFL');
    
    % 设置标签、标题和图例
    xlabel('Epoch', 'FontSize', 12);
    ylabel('Accuracy', 'FontSize', 12);
    %title('Mean Accuracy with 95% Confidence Intervals for DLora & Lora', 'FontSize', 14);
    % 设置图例，并允许自定义位置
    legend('show', 'Location', 'southeast');
    
    grid on;
    xlim([1 260]);
    ylim([0.4 0.95])

end


plot_accuracies_with_confidence_intervals(mean_accuracies1, confidence_intervals1,mean_accuracies2, confidence_intervals2)
