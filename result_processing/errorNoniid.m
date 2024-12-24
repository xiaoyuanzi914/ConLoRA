% 定义数据
x = 1:256;  % x 轴数据，例如 1 到 10
% 数据
DLoRA_iid = [];
LoRA_DFL_iid = [];
DLoRA_noniid = [];
LoRA_DFL_noniid = [];
% 绘制折线图
figure;
plot(x, DLoRA_iid, '-', 'LineWidth', 1.5); % 第一组数据，实线
hold on;
plot(x, LoRA_DFL_iid, '-', 'LineWidth', 1.5); % 第二组数据，虚线
plot(x, DLoRA_noniid, '-', 'LineWidth', 1.5); % 第三组数据，点线
plot(x, LoRA_DFL_noniid, '-', 'LineWidth', 1.5); % 第四组数据，点划线
hold off;

% 设置图例
legend('ConLoRA-IID', 'LoRA-DFL-IID', 'ConLoRA-nonIID', 'LoRA-DFL-nonIID',Location='northwest');

% 设置标题和轴标签
%title('Line Plot of Four Data Groups');
xlabel('Epoch');
ylabel('Consensus error');

% 添加网格
grid on;

% 调整轴范围
xlim([1 260]);
ylim([0 22]);
