function data_analyzer(filename)
    % 读取CSV文件
    data = readtable(filename);
    
    % 获取所有列名（除了时间戳列）
    column_names = data.Properties.VariableNames;
    data_columns = column_names(2:end);  % 跳过时间戳列
    
    % 获取时间戳
    timestamps = data.Timestamp_ms_;
    time_interval = mean(diff(timestamps)) / 1000;  % 转换为秒
    
    % 创建颜色映射
    colors = lines(length(data_columns));  % 使用MATLAB的lines颜色映射
    
    % 1. 时域分析图
    figure('Name', '时域分析', 'NumberTitle', 'off');
    hold on;
    for i = 1:length(data_columns)
        col_name = data_columns{i};
        values = data.(col_name);
        plot(timestamps/1000, values, 'Color', colors(i,:), 'LineWidth', 1.5);
    end
    hold off;
    title('时域波形');
    xlabel('时间 (s)');
    ylabel('幅值');
    legend(data_columns, 'Location', 'best');
    grid on;
    
    % 2. 频域分析图
    figure('Name', '频域分析', 'NumberTitle', 'off');
    hold on;
    for i = 1:length(data_columns)
        col_name = data_columns{i};
        values = data.(col_name);
        
        % 计算FFT
        N = length(values);
        fft_values = fft(values);
        fft_values = abs(fft_values(1:N/2+1));
        frequencies = (0:N/2) / (N * time_interval);
        
        plot(frequencies, fft_values, 'Color', colors(i,:), 'LineWidth', 1.5);
    end
    hold off;
    title('频域分析');
    xlabel('频率 (Hz)');
    ylabel('幅值');
    legend(data_columns, 'Location', 'best');
    grid on;
    
    % 3. 统计信息表格
    figure('Name', '统计信息', 'NumberTitle', 'off');
    % 准备统计信息
    stats = cell(length(data_columns), 6);
    for i = 1:length(data_columns)
        values = data.(data_columns{i});
        stats(i,:) = {
            data_columns{i}, ...
            mean(values), ...
            std(values), ...
            min(values), ...
            max(values), ...
            median(values)
        };
    end
    
    % 创建表格
    t = uitable('Data', stats, ...
        'ColumnName', {'数据名称', '平均值', '标准差', '最小值', '最大值', '中位数'}, ...
        'RowName', [], ...
        'Position', [20 20 600 150]);
    
    % 显示数据基本信息
    fprintf('数据基本信息:\n');
    fprintf('总采样点数: %d\n', length(timestamps));
    fprintf('采样间隔: %.3f ms\n', time_interval * 1000);
    fprintf('采样率: %.2f Hz\n', 1/time_interval);
    fprintf('数据列: %s\n', strjoin(data_columns, ', '));
end
