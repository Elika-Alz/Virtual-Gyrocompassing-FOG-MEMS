%% Batch evaluation of STIM headings (Independent mode)
% FOG is only used here as ground truth at the end.

clearvars; close all; clc;

%% Config
TAU_OPT_SEC = 32.6; 

BIAS_INSTABILITY = 0.000184; 
KF_PROCESS_NOISE = (BIAS_INSTABILITY)^2;  
KF_MEASURE_NOISE = 1e-2;  

R_FOG_TO_PLATFORM  = diag([1, -1, -1]);
R_STIM_TO_PLATFORM = eye(3);
FOG_OUTLIER_SIGMA = 7;

%% Load files
disp('Select IMU test files (you can pick multiple)...');
[fileNames, dataPath] = uigetfile('*.mat', 'Select files', 'MultiSelect', 'on');

if isequal(fileNames, 0), error('Canceled.'); end
if ischar(fileNames), fileNames = {fileNames}; end 

numFiles = length(fileNames);
final_fog_headings = zeros(numFiles, 1);
final_stim_headings = zeros(numFiles, 1);
final_errors = zeros(numFiles, 1);
file_labels = cell(numFiles, 1);

time_cell = cell(numFiles, 1);
trend_cell = cell(numFiles, 1);

fprintf('Evaluating %d files...\n\n', numFiles);

%% Process loop
for i = 1:numFiles
    fprintf('Test %d: %s\n', i, fileNames{i});
    file_labels{i} = sprintf('Test %d', i);
    
    loadedData = load(fullfile(dataPath, fileNames{i}), "navDataArray", "stimDataArray");
    fogData  = double(loadedData.navDataArray(:, 1:4));
    
    rawStim = double(loadedData.stimDataArray);
    if size(rawStim, 2) >= 7
        stimData = rawStim(:, 1:7); hasAccel = true;
    else
        stimData = rawStim(:, 1:4); hasAccel = false;
    end
    
    % sync
    startTime = max(fogData(1,1), stimData(1,1));
    maxTime = min(fogData(end,1), stimData(end,1)) - startTime;
    
    fMask = fogData(:,1) >= startTime & fogData(:,1) <= (startTime + maxTime);
    sMask = stimData(:,1) >= startTime & stimData(:,1) <= (startTime + maxTime);
    
    fogWindow = fogData(fMask, :); stimWindow = stimData(sMask, :);
    plotTimeFOG = fogWindow(:,1) - startTime; plotTimeSTIM = stimWindow(:,1) - startTime;
    
    % leveling
    if hasAccel
        mean_acc = mean(stimWindow(:, 5:7)); 
        acc_platform = (R_STIM_TO_PLATFORM * mean_acc.').';
        g_vec = acc_platform / norm(acc_platform);
        v = cross(g_vec', [0; 0; 1]); c = dot(g_vec', [0; 0; 1]); s = norm(v);
        if s < 1e-6, R_level = eye(3); else
            Vx = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
            R_level = eye(3) + Vx + Vx^2 * ((1 - c) / s^2);
        end
    else
        R_level = eye(3); 
    end
    
    % calc FOG baseline
    fogInterval = median(diff(fogWindow(:, 1)));
    fogRatePlatform = (R_FOG_TO_PLATFORM * (fogWindow(:, 2:4) / fogInterval).').';
    fogRateLeveled = (R_level * fogRatePlatform.').';
    
    fogMask = robustThreeAxisMask(fogRateLeveled, FOG_OUTLIER_SIGMA);
    fGood = fogRateLeveled(fogMask, :); plotTimeFOG_clean = plotTimeFOG(fogMask);
    
    fogStableMask = plotTimeFOG_clean >= (maxTime - TAU_OPT_SEC);
    fog_overall_meanX = mean(fGood(fogStableMask, 1));
    fog_overall_meanY = mean(fGood(fogStableMask, 2));
    fog_heading = mod(atan2d(fog_overall_meanY, fog_overall_meanX), 360);
    
    % calc STIM (independent)
    stimRatePlatform = (R_STIM_TO_PLATFORM * stimWindow(:, 2:4).').';
    stimRateLeveled = (R_level * stimRatePlatform.').';
    
    % no transfer alignment here, just raw leveled STIM
    stimRateCorrected = stimRateLeveled; 
    
    % run KF
    N = size(stimRateCorrected, 1);
    kf_est_X = zeros(N, 1); kf_est_Y = zeros(N, 1);
    P_x = 1; P_y = 1; x_est = stimRateCorrected(1, 1); y_est = stimRateCorrected(1, 2);
    
    for k = 1:N
        P_x = P_x + KF_PROCESS_NOISE; P_y = P_y + KF_PROCESS_NOISE;
        K_x = P_x / (P_x + KF_MEASURE_NOISE); K_y = P_y / (P_y + KF_MEASURE_NOISE);
        x_est = x_est + K_x * (stimRateCorrected(k, 1) - x_est);
        y_est = y_est + K_y * (stimRateCorrected(k, 2) - y_est);
        P_x = (1 - K_x) * P_x; P_y = (1 - K_y) * P_y;
        kf_est_X(k) = x_est; kf_est_Y(k) = y_est;
    end
    
    last_tau_idx = find(plotTimeSTIM > (plotTimeSTIM(end) - TAU_OPT_SEC), 1);
    stim_final_X = mean(kf_est_X(last_tau_idx:end)); 
    stim_final_Y = mean(kf_est_Y(last_tau_idx:end));
    stim_heading = mod(atan2d(stim_final_Y, stim_final_X), 360);
    
    % save errors
    diffDeg = stim_heading - fog_heading;
    diffDeg = mod(diffDeg + 180, 360) - 180; 
    
    final_fog_headings(i) = fog_heading;
    final_stim_headings(i) = stim_heading;
    final_errors(i) = diffDeg;
    
    % prep trend data
    actual_stim_freq = length(plotTimeSTIM) / (plotTimeSTIM(end) - plotTimeSTIM(1));
    win_stim_samples = round(TAU_OPT_SEC * actual_stim_freq);
    
    stim_runX = movmean(kf_est_X, [win_stim_samples-1, 0]);
    stim_runY = movmean(kf_est_Y, [win_stim_samples-1, 0]);
    
    % start plotting after 60s (initial warm-up)
    valid_idx = plotTimeSTIM > 60; 
    time_cell{i} = plotTimeSTIM(valid_idx) / 60;
    
    stim_cont = mod(atan2d(stim_runY(valid_idx), stim_runX(valid_idx)), 360);
    diff_cont = stim_cont - fog_heading;
    trend_cell{i} = mod(diff_cont + 180, 360) - 180;
end

%% Print summary
fprintf('\n-- Final Summary --\n');
for i = 1:numFiles
    fprintf('%s | FOG: %7.2f | STIM: %7.2f | Err: %+7.2f\n', ...
        file_labels{i}, final_fog_headings(i), final_stim_headings(i), final_errors(i));
end

%% Plot
figure('Name', 'Independent Trends', 'NumberTitle', 'off', 'Position', [150 150 1000 500]);
hold on; grid on;
colors = lines(numFiles);

for i = 1:numFiles
    plot(time_cell{i}, trend_cell{i}, 'Color', colors(i,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (Err: %+.1f)', file_labels{i}, final_errors(i)));
end

yline(0, 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xLimits = xlim; yLimits = ylim;
patch([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [5 5 -5 -5], 'g', ...
      'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '5 deg target zone');
      
ylim([min(-6, yLimits(1)-2), max(6, yLimits(2)+2)]); 
xlabel('Time (mins)');
ylabel('Heading Diff (deg)');
title('STIM Independent Trends vs FOG');
legend('Location', 'best'); xlim(xLimits);

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