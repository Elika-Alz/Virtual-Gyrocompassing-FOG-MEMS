%% Gyrocompassing test optimized with Allan Variance params
% tau_opt is around 32.6s based on the AV plot

clearvars; close all; clc;

%% Params
LATITUDE_DEG = 40.84;
TARGET_ALIGNMENT_SEC = 300; % 5 mins total test time

% optimal averaging time
TAU_OPT_SEC = 32.6; 

% transfer alignment window (start after 3 mins to let it settle)
TRANSFER_START_SEC = 180;                         
TRANSFER_END_SEC   = TRANSFER_START_SEC + TAU_OPT_SEC; 

% KF settings (tuned from bias instability)
BIAS_INSTABILITY = 0.000184; % deg/s (Y-axis AV)
KF_PROCESS_NOISE = (BIAS_INSTABILITY)^2;  
KF_MEASURE_NOISE = 1e-2;  

% coord transforms (X=North, Y=West, Z=Up)
R_FOG_TO_PLATFORM  = diag([1, -1, -1]);
R_STIM_TO_PLATFORM = eye(3);

FOG_OUTLIER_SIGMA = 7;

%% Load data
disp('--- Allan-optimized transfer alignment ---');
[dataFileName, dataPathName] = uigetfile('*.mat', 'Select 5-min IMU file');
if isequal(dataFileName, 0), error('Canceled by user.'); end

loadedData = load(fullfile(dataPathName, dataFileName), "navDataArray", "stimDataArray");
fogData  = double(loadedData.navDataArray(:, 1:4));
rawStim = double(loadedData.stimDataArray);

if size(rawStim, 2) >= 7
    stimData = rawStim(:, 1:7);
    hasAccel = true;
else
    stimData = rawStim(:, 1:4);
    hasAccel = false;
    disp('Warning: STIM data has 4 cols. Skipping virtual leveling.');
end

%% Sync times
startTime = max(fogData(1,1), stimData(1,1));
maxAvailableTime = min(fogData(end,1), stimData(end,1)) - startTime;
actualProcessTime = min(TARGET_ALIGNMENT_SEC, maxAvailableTime);

fMask = fogData(:,1) >= startTime & fogData(:,1) <= (startTime + actualProcessTime);
sMask = stimData(:,1) >= startTime & stimData(:,1) <= (startTime + actualProcessTime);

fogWindow  = fogData(fMask, :);
stimWindow = stimData(sMask, :);

plotTimeFOG = fogWindow(:,1) - startTime;
plotTimeSTIM = stimWindow(:,1) - startTime;

%% Virtual leveling
disp('Step 1: Calc platform tilt...');

if hasAccel
    mean_acc = mean(stimWindow(:, 5:7)); 
    acc_platform = (R_STIM_TO_PLATFORM * mean_acc.').';

    g_vec = acc_platform / norm(acc_platform);
    target_g = [0; 0; 1]; 

    v = cross(g_vec', target_g);
    c = dot(g_vec', target_g);
    s = norm(v);

    if s < 1e-6
        R_level = eye(3);
    else
        Vx = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
        R_level = eye(3) + Vx + Vx^2 * ((1 - c) / s^2);
    end
else
    R_level = eye(3); 
end

%% FOG processing (ground truth)
disp('Step 2: Process FOG data...');
fogInterval = median(diff(fogWindow(:, 1)));
fogRatePlatform = (R_FOG_TO_PLATFORM * (fogWindow(:, 2:4) / fogInterval).').';
fogRateLeveled = (R_level * fogRatePlatform.').';

fogMask = robustThreeAxisMask(fogRateLeveled, FOG_OUTLIER_SIGMA);
fGood = fogRateLeveled(fogMask, :);
plotTimeFOG_clean = plotTimeFOG(fogMask);

% grab baseline from the last tau_opt window
fogStableMask = plotTimeFOG_clean >= (actualProcessTime - TAU_OPT_SEC);
fog_overall_meanX = mean(fGood(fogStableMask, 1));
fog_overall_meanY = mean(fGood(fogStableMask, 2));

fog_heading = mod(atan2d(fog_overall_meanY, fog_overall_meanX), 360);

%% Transfer alignment
fprintf('Step 3: Run transfer alignment (Window: %.1fs)...\n', TAU_OPT_SEC);

fTransferMask = plotTimeFOG_clean >= TRANSFER_START_SEC & plotTimeFOG_clean <= TRANSFER_END_SEC;
fog_transfer_meanX = mean(fGood(fTransferMask, 1));
fog_transfer_meanY = mean(fGood(fTransferMask, 2));

stimRatePlatform = (R_STIM_TO_PLATFORM * stimWindow(:, 2:4).').';
stimRateLeveled  = (R_level * stimRatePlatform.').';

sTransferMask = plotTimeSTIM >= TRANSFER_START_SEC & plotTimeSTIM <= TRANSFER_END_SEC;
stim_raw_transferX = mean(stimRateLeveled(sTransferMask, 1));
stim_raw_transferY = mean(stimRateLeveled(sTransferMask, 2));

% calc real-time bias
dynamic_bias_X = stim_raw_transferX - fog_transfer_meanX;
dynamic_bias_Y = stim_raw_transferY - fog_transfer_meanY;

%% Kalman filter
disp('Step 4: Run KF...');

stimRateCorrected = stimRateLeveled;
stimRateCorrected(:, 1) = stimRateCorrected(:, 1) - dynamic_bias_X;
stimRateCorrected(:, 2) = stimRateCorrected(:, 2) - dynamic_bias_Y;

N = size(stimRateCorrected, 1);
kf_est_X = zeros(N, 1);
kf_est_Y = zeros(N, 1);

P_x = 1; P_y = 1; 
x_est = stimRateCorrected(1, 1); 
y_est = stimRateCorrected(1, 2);

for k = 1:N
    x_pred = x_est; y_pred = y_est;
    P_x = P_x + KF_PROCESS_NOISE; P_y = P_y + KF_PROCESS_NOISE;
    
    K_x = P_x / (P_x + KF_MEASURE_NOISE); K_y = P_y / (P_y + KF_MEASURE_NOISE);
    
    x_est = x_pred + K_x * (stimRateCorrected(k, 1) - x_pred);
    y_est = y_pred + K_y * (stimRateCorrected(k, 2) - y_pred);
    
    P_x = (1 - K_x) * P_x; P_y = (1 - K_y) * P_y;
    
    kf_est_X(k) = x_est; kf_est_Y(k) = y_est;
end

last_tau_idx = find(plotTimeSTIM > (plotTimeSTIM(end) - TAU_OPT_SEC), 1);
if isempty(last_tau_idx), last_tau_idx = floor(N * 0.8); end

stim_final_X = mean(kf_est_X(last_tau_idx:end)); 
stim_final_Y = mean(kf_est_Y(last_tau_idx:end));

stim_heading = mod(atan2d(stim_final_Y, stim_final_X), 360);

%% Results
diffDeg = stim_heading - fog_heading;
diffDeg = mod(diffDeg + 180, 360) - 180; 

fprintf('\n-- Results --\n');
fprintf('FOG baseline : %8.3f deg\n', fog_heading);
fprintf('STIM virtual : %8.3f deg\n', stim_heading);
fprintf('Diff         : %+8.3f deg\n', diffDeg);

figure('Name', 'Alignment Results', 'NumberTitle', 'off', 'Position', [100 100 1200 500]);

subplot(1,2,1);
plot(plotTimeFOG_clean, fGood(:,1), 'Color', [0.8 0.9 0.8], 'DisplayName', 'FOG X'); hold on;
plot(plotTimeSTIM, stimRateCorrected(:,1), 'Color', [0.8 0.8 0.8], 'DisplayName', 'STIM raw X');
plot(plotTimeSTIM, kf_est_X, 'b', 'LineWidth', 1.5, 'DisplayName', 'STIM KF X');
xline(TRANSFER_START_SEC, 'k:', 'Transfer start', 'LineWidth', 1.5, 'HandleVisibility','off');
xline(TRANSFER_END_SEC, 'k:', 'Transfer end', 'LineWidth', 1.5, 'HandleVisibility','off');
yline(fog_overall_meanX, 'g--', 'FOG Base', 'LineWidth', 1.5);
yline(stim_final_X, 'r--', 'STIM final', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Rate (deg/s)'); title('X-Axis'); 
legend('Location', 'best'); grid on; axis tight;

subplot(1,2,2);
plot(plotTimeFOG_clean, fGood(:,2), 'Color', [0.8 0.9 0.8], 'DisplayName', 'FOG Y'); hold on;
plot(plotTimeSTIM, stimRateCorrected(:,2), 'Color', [0.8 0.8 0.8], 'DisplayName', 'STIM raw Y');
plot(plotTimeSTIM, kf_est_Y, 'b', 'LineWidth', 1.5, 'DisplayName', 'STIM KF Y');
xline(TRANSFER_START_SEC, 'k:', 'Transfer start', 'LineWidth', 1.5, 'HandleVisibility','off');
xline(TRANSFER_END_SEC, 'k:', 'Transfer end', 'LineWidth', 1.5, 'HandleVisibility','off');
yline(fog_overall_meanY, 'g--', 'FOG Base', 'LineWidth', 1.5);
yline(stim_final_Y, 'r--', 'STIM final', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Rate (deg/s)'); title('Y-Axis'); 
legend('Location', 'best'); grid on; axis tight;

sgtitle(sprintf('Alignment | FOG: %.2f° | STIM: %.2f° | Diff: %+.2f°', fog_heading, stim_heading, diffDeg), 'FontWeight', 'bold');

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