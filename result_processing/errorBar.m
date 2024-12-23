% 定义数据
network_density = [3, 4, 5, 6];
error_values_group1 = [12.42, 9.84, 5.37, 4.29];
error_values_group2 = [1.70, 1.38, 0.82, 0.53];

% 将两组数据组合成一个矩阵
error_values = [error_values_group1; error_values_group2]';

% 绘制分组柱状图
figure;
b = bar(network_density, error_values, 'grouped'); % 返回柱状图对象

% 设置轴标签
xlabel('Average node connectivity');
ylabel('Consensus error');

% 添加图例
legend('LoRA-DFL', 'ConLoRA');

% 显示网格
grid on;

% 在每个柱子上显示数值
% 获取柱子的x坐标和高度
for i = 1:length(b)
    xData = b(i).XEndPoints; % 获取每组柱子的x坐标
    yData = b(i).YData; % 获取柱子的y值
    for j = 1:length(xData)
        text(xData(j), yData(j), sprintf('%.2f', yData(j)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end
