% MATLAB Script to Plot Tello IMU Data from CSV

% --- Step 1: Select the CSV file ---
% Use uigetfile to allow the user to browse and select the data file
[filename, pathname] = uigetfile('*.csv', 'Select the Tello IMU data CSV file');

% Check if the user cancelled the file selection
if isequal(filename, 0) || isequal(pathname, 0)
    disp('User cancelled file selection. Exiting.');
    return; % Exit the script
else
    % Construct the full file path
    fullFilePath = fullfile(pathname, filename);
    disp(['Loading data from: ', fullFilePath]);
end

% --- Step 2: Read the data from the CSV file ---
try
    % Use readtable for easy handling of headers
    % It assumes the header names are exactly 'Yaw', 'Pitch', 'Roll'
    dataTable = readtable(fullFilePath);

    % Extract the data columns by their header names
    yaw = dataTable.Yaw;
    pitch = dataTable.Pitch;
    roll = dataTable.Roll;

    % Determine the number of data points (samples)
    numSamples = height(dataTable);

    % Create a vector representing the sample number (or time steps if sampling rate is known)
    sampleIndex = 1:numSamples; % Simple index for plotting order

    % --- Step 3: Plot the data ---
    figure; % Create a new figure window
    hold on; % Keep the plot active to add multiple lines

    % Plot Yaw vs Sample Index
    plot(sampleIndex, yaw, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Yaw');

    % Plot Pitch vs Sample Index
    plot(sampleIndex, pitch, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Pitch');

    % Plot Roll vs Sample Index
    plot(sampleIndex, roll, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Roll');

    hold off; % Release the plot hold

    % --- Step 4: Add labels, title, legend, and grid ---
    title('Tello IMU Data Over Time');
    xlabel('Sample Index');
    % Assuming the Tello library returns angles in degrees
    ylabel('Angle (degrees)');
    legend('show', 'Location', 'best'); % Show the legend based on DisplayName
    grid on; % Turn on the grid for easier reading

    disp('Data plotted successfully.');

catch ME
    % Catch potential errors during file reading or plotting
    fprintf('Error processing file: %s\n', ME.message);
    if strcmp(ME.identifier, 'MATLAB:readtable:CannotOpenFile')
        disp(['Error: Could not open the specified file: "', fullFilePath, '".']);
        disp('Please ensure the file exists and is not corrupted.');
    elseif contains(ME.identifier, 'UnrecognizedVariable') || contains(ME.identifier, 'UndefinedFunctionOrVariable')
        disp('Error: Could not find expected column headers (Yaw, Pitch, Roll) in the CSV file.');
        disp('Please check the CSV file header row.');
    else
        disp('An unexpected error occurred. Check file format and MATLAB environment.');
    end
end

