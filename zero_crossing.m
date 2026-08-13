%% Find the empirical zero-crossing (golden window)
% Checks where STIM and FOG headings match best.

clearvars; close all; clc;

%% Config
TAU_OPT_SEC = 32.6; % from Allan Variance
FOG_OUTLIER_SIGMA = 7;
R_FOG_TO_PLATFORM  = diag([1, -1, -1]);
R_STIM_TO_PLATFORM = eye(3);

disp('--- Zero-crossing finder ---');
disp('Select all IMU test files...');
[fileNames, dataPath] = uigetfile('*.mat', 'Select files', 'MultiSelect', 'on');

if isequal(fileNames, 0), error('Canceled.'); end
if ischar(fileNames), fileNames = {fileNames}; end 

numFiles = length(fileNames);
best_times_sec = zeros(numFiles, 1);
min_errors_deg = zeros(numFiles, 1);

fprintf('Found %d files. Analyzing...\n\n', numFiles);

%% Batch loop
for i = 1:numFiles
    fprintf('File %d/%d: %s... ', i, numFiles, fileNames{i});
    
    loadedData = load(fullfile(dataPath, fileNames{i}), "navDataArray", "stimDataArray");
    fogData  = double(loadedData.navDataArray(:, 1:4));
    
    rawStim = double(loadedData.stimDataArray);
    if size(rawStim, 2) >= 7
        stimData = rawStim(:, 1:7);
        mean_acc = mean(stimData(:, 5:7)); 
        acc_platform = (R_STIM_TO_PLATFORM * mean_acc.').';
        g_vec = acc_platform / norm(acc_platform);
        target_g = [0; 0; 1]; 
        v = cross(g_vec', target_g); c = dot(g_vec', target_g); s = norm(v);
        if s < 1e-6, R_level = eye(3); else
            Vx = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
            R_level = eye(3) + Vx + Vx^2 * ((1 - c) / s^2);
        end
    else
        stimData = rawStim(:, 1:4);
        R_level = eye(3); 
    end
    
    startTime = max(fogData(1,1), stimData(1,1));
    maxTime = min(fogData(end,1), stimData(end,1)) - startTime;
    
    fMask = fogData(:,1) >= startTime & fogData(:,1) <= (startTime + maxTime);
    sMask = stimData(:,1) >= startTime & stimData(:,1) <= (startTime + maxTime);
    
    fogWindow = fogData(fMask, :);
    stimWindow = stimData(sMask, :);
    plotTimeSTIM = stimWindow(:,1) - startTime;
    plotTimeFOG = fogWindow(:,1) - startTime;
    
    % process rates
    fogInterval = median(diff(fogWindow(:, 1)));
    fogRatePlatform = (R_FOG_TO_PLATFORM * (fogWindow(:, 2:4) / fogInterval).').';
    fogRateLeveled = (R_level * fogRatePlatform.').';
    
    fogMask = robustThreeAxisMask(fogRateLeveled, FOG_OUTLIER_SIGMA);
    fGood = fogRateLeveled(fogMask, :);
    plotTimeFOG_clean = plotTimeFOG(fogMask);
    
    stimRatePlatform = (R_STIM_TO_PLATFORM * stimWindow(:, 2:4).').';
    stimRateLeveled  = (R_level * stimRatePlatform.').';
    
    % sliding windows
    stimInterval = median(diff(plotTimeSTIM));
    win_fog_samples = round(TAU_OPT_SEC / fogInterval);
    win_stim_samples = round(TAU_OPT_SEC / stimInterval);
    
    fog_runX = movmean(fGood(:,1), [win_fog_samples-1, 0]);
    fog_runY = movmean(fGood(:,2), [win_fog_samples-1, 0]);
    fog_heading = mod(atan2d(fog_runY, fog_runX), 360);
    
    stim_runX = movmean(stimRateLeveled(:,1), [win_stim_samples-1, 0]);
    stim_runY = movmean(stimRateLeveled(:,2), [win_stim_samples-1, 0]);
    stim_heading = mod(atan2d(stim_runY, stim_runX), 360);
    
    % sync sizes
    fog_unwrapped = unwrap(fog_heading * pi/180) * 180/pi;
    fog_interp = interp1(plotTimeFOG_clean, fog_unwrapped, plotTimeSTIM, 'linear', 'extrap');
    fog_interp_wrapped = mod(fog_interp, 360);
    
    diff_trend = stim_heading - fog_interp_wrapped;
    diff_trend = mod(diff_trend + 180, 360) - 180;
    
    % skip the first 60s (thermal shock/settling)
    valid_idx = plotTimeSTIM > 60;
    valid_time = plotTimeSTIM(valid_idx);
    valid_diff = abs(diff_trend(valid_idx));
    
    % find min error
    [min_err, min_idx] = min(valid_diff);
    best_time = valid_time(min_idx);
    
    best_times_sec(i) = best_time;
    min_errors_deg(i) = min_err;
    
    fprintf('Best time: %.1fs (Err: %.2f)\n', best_time, min_err);
    
    clear loadedData fogData stimData rawStim;
end

%% Stats & plot
average_golden_time = mean(best_times_sec);
median_golden_time = median(best_times_sec);

fprintf('\n-- Final Results --\n');
fprintf('Mean zero-cross time   : %.1f sec (%.1f mins)\n', average_golden_time, average_golden_time/60);
fprintf('Median zero-cross time : %.1f sec (%.1f mins)\n', median_golden_time, median_golden_time/60);

figure('Name', 'Golden Window', 'NumberTitle', 'off', 'Position', [200 200 800 500]);
scatter(1:numFiles, best_times_sec, 100, 'b', 'filled', 'MarkerEdgeColor', 'k'); hold on;
yline(average_golden_time, 'r-', 'Mean', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
yline(median_golden_time, 'g--', 'Median', 'LineWidth', 2, 'LabelHorizontalAlignment', 'right');

grid on;
xticks(1:numFiles);
xlabel('Test File');
ylabel('Zero-Cross Time (s)');
title('Zero-Crossing Analysis');
legend('Zero-cross point', 'Average', 'Median', 'Location', 'best');

function good = robustThreeAxisMask(rateData, sigmaLimit)
    good = all(isfinite(rateData), 2);
    for axisNumber = 1:3
        axisData = rateData(:, axisNumber);
        axisMedian = median(axisData);
        robustSigma = 1.4826 * median(abs(axisData - axisMedian));
        if isfinite(robustSigma) && robustSigma > 0
            good = good & abs(axisData - axisMedian) <= sigmaLimit * robustSigma;
        end
    end
end