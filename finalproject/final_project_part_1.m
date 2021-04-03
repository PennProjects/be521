%% Final project part 1
% Prepared by John Bernabei and Brittany Scheid

% One of the oldest paradigms of BCI research is motor planning: predicting
% the movement of a limb using recordings from an ensemble of cells involved
% in motor control (usually in primary motor cortex, often called M1).

% This final project involves predicting finger flexion using intracranial EEG (ECoG) in three human
% subjects. The data and problem framing come from the 4th BCI Competition. For the details of the
% problem, experimental protocol, data, and evaluation, please see the original 4th BCI Competition
% documentation (included as separate document). The remainder of the current document discusses
% other aspects of the project relevant to BE521.


%% Start the necessary ieeg.org sessions (0 points)

%fetching data from the ieee server
addpath(genpath('/Users/jalpanchal/git/be521'));

username = 'jalpanchal';
passPath = 'jal_ieeglogin.bin';

% Load training ecog from each of three patients
s1_train_ecog_session = IEEGSession('I521_Sub1_Training_ecog', username, passPath);
s2_train_ecog_session = IEEGSession('I521_Sub2_Training_ecog', username, passPath);
s3_train_ecog_session = IEEGSession('I521_Sub3_Training_ecog', username, passPath);


% Load training dataglove finger flexion values for each of three patients
s1_train_dg_session = IEEGSession('I521_Sub1_Training_dg', username, passPath);
s2_train_dg_session = IEEGSession('I521_Sub2_Training_dg', username, passPath);
s3_train_dg_session = IEEGSession('I521_Sub3_Training_dg', username, passPath);

%% Extract dataglove and ECoG data 
% Dataglove should be (samples x 5) array 
% ECoG should be (samples x channels) array

% Split data into a train and test set (use at least 50% for training)

%load data from mat file
load('final_proj_part1_data.mat')
%This data has 61 ch for s1,46 ch for s2 and 64 ch for s3
%training set contain 75% of 300s data
%ecog data
s1_train_ecog_cleaned = train_ecog{1}(1:225000,:);
s2_train_ecog_cleaned = train_ecog{2}(1:225000,:);
s3_train_ecog_cleaned = train_ecog{3}(1:225000,:);
%finger data
s1_train_dg = train_dg{1}(1:225000,:);
s2_train_dg = train_dg{2}(1:225000,:);
s3_train_dg = train_dg{3}(1:225000,:);

%test set containing last 25% of 300 data
%ecog data
s1_test_ecog_cleaned = train_ecog{1}(225001:300000,:);
s2_test_ecog_cleaned = train_ecog{2}(225001:300000,:);
s3_test_ecog_cleaned = train_ecog{3}(225001:300000,:);
%finger dta
s1_test_dg = train_dg{1}(225001:300000,:);
s2_test_dg = train_dg{2}(225001:300000,:);
s3_test_dg = train_dg{3}(225001:300000,:);

%%
% testing filter
% fs = 1000;                    % Sampling frequency (samples per second)
% dt = 1/fs;                   % seconds per sample
% StopTime = 0.25;             % seconds
% t = (0:dt:StopTime-dt)';     % seconds
% F = 60;                      % Sine wave frequency (hertz)
% data = sin(2*pi*F*t);
% 
% F2  = 300
% data2 = sin(2*pi*F2*t);
% 
% x= data + data2;
% subplot(2,1,1)
% plot(x);
% 
% 
% y = filter_data(x)
% subplot(2,1,2)
% plot(y)



%%
% \textbf{Answer 1.1} \\
% The number of samples in the ECoG recording for each of the 3 subjects is
% 300,000. This consists of 300s of data sampled at 1000 Hz. Yes, it is the
% same for all three subjects.
% \textbf{Answer 1.2} \\
% I used a bandpass filter, with a passband between 0.15 and 200 Hz.


%% Get Features
% run getWindowedFeats_release function

NumWins = @(xLen, winLen, winOverlap) floor((xLen-(winOverlap))/(winLen-winOverlap));

winLen_ms = 100;
winOverlap_ms = 50;
s1_length_ms = 300000;
s1_number_win = NumWins(s1_length_ms, winLen_ms,winOverlap_ms)

%%
% \textbf{Answer 2.1} \\
% For a Window length of 100ms and a 50ms overlap we will have 5999
% windows in the 300s data for each subject. The 75% taken for training has
% 4499 windows and the 25 % test set has 1499 windows.
% \textbf{Answer 2.2} \\
% Implemented in get\_features.m. The features calculated are : Line length,
% Area, Energy, Mean Voltage (LMP)
% \textbf{Answer 2.3} \\
% Implemented in get\_windowedFeats.m
% \textbf{Answer 3.1} \\
% The dimension of R matrix for subject 1 would be 5999 x 1117 when we include the 1's columns.
% Else it would be 5999 x 1116. Number of columns  = 1+N*6*62
% = 1117. Number of rows = number of time windows = 5999.


%% Create R matrix

% \textbf{Answer 3.2} \\
% Testing create\_r\_matrix with testRFunction.mat gives a value of 25.4668
load('testRfunction.mat')
R = create_R_matrix(testR_features,N_wind);
test_R = mean(mean(R))

%%
%Getting R matrix for each subject
fs_hz = 1000;
win_length_ms  = 100;
win_overlap_ms = 50;
n_wind = 3;

%subject 1
s1_window_feats = getWindowedFeats(s1_train_ecog_cleaned, fs_hz, win_length_ms, win_overlap_ms);
s1_R_train = create_R_matrix(s1_window_feats, n_wind);

disp('sub1 completed')
%subject 2
s2_window_feats = getWindowedFeats(s2_train_ecog_cleaned, fs_hz, win_length_ms, win_overlap_ms);
s2_R_train = create_R_matrix(s2_window_feats, n_wind);

disp('sub2 completed')
%subject 3
s3_window_feats = getWindowedFeats(s3_train_ecog_cleaned, fs_hz, win_length_ms, win_overlap_ms);
s3_R_train = create_R_matrix(s3_window_feats, n_wind);


%% Train classifiers (8 points)



% Classifier 1: Get angle predictions using optimal linear decoding. That is, 
% calculate the linear filter (i.e. the weights matrix) as defined by 
% Equation 1 for all 5 finger angles.



%%
%first we will create the target matrix y for each subject
%downsampling dg data to math windows in R matrix
%selecting a point every 50ms
s1_y_train = downsample(s1_train_dg,50);
s1_y_train = s1_y_train(1:end-1,:); %reducing from 4500 to 4499 samples
s2_y_train = downsample(s2_train_dg,50);
s2_y_train = s2_y_train(1:end-1,:);
s3_y_train = downsample(s3_train_dg,50);
s3_y_train = s3_y_train(1:end-1,:);

%%
%calculating f matrix

s1_f = (s1_R_train'*s1_R_train)\(s1_R_train'*s1_y_train);
s2_f = (s2_R_train'*s2_R_train)\(s2_R_train'*s2_y_train);
s3_f = (s3_R_train'*s3_R_train)\(s3_R_train'*s3_y_train);

%%
%calculating training error
s1_train_pred_y = s1_R_train*s1_f;
s2_train_pred_y = s2_R_train*s2_f;
s3_train_pred_y = s3_R_train*s3_f;

%upsampling training pred y to 1000 hz signal
s1_train_pred_y_upsamp = zoh_upsample(s1_train_pred_y,50);
s2_train_pred_y_upsamp = zoh_upsample(s2_train_pred_y,50);
s3_train_pred_y_upsamp = zoh_upsample(s3_train_pred_y,50);

%calculating training correlation coeff
s1_train_rho = corr(s1_train_pred_y_upsamp, s1_train_dg);
s1_train_rho = diag(s1_train_rho)'

s2_train_rho = corr(s2_train_pred_y_upsamp, s2_train_dg);
s2_train_rho = diag(s2_train_rho)'

s3_train_rho = corr(s3_train_pred_y_upsamp, s3_train_dg);
s3_train_rho = diag(s3_train_rho)'

% Try at least 1 other type of machine learning algorithm, you may choose
% to loop through the fingers and train a separate classifier for angles 
% corresponding to each finger


%%
%To build a ML model we will first create new features from 100 ms before
%the activity
%Averaging features from M, M-1 and M-2 windows
s1_window_feats_avg = avg_features(s1_window_feats,3);
s2_window_feats_avg = avg_features(s2_window_feats,3);
s3_window_feats_avg = avg_features(s3_window_feats,3);

%%

X = s1_window_feats;
Y = s1_y_train(:,1);
svmodel_ = fitcsvm(X,Y, 'KernelFunction','rbf');
Ypred_train_svm = predict(svmodel, X);
train_error_svm = size(find(Ypred_train_svm~=Y),1)/size(Y,1)




% Try a form of either feature or prediction post-processing to try and
% improve underlying data or predictions.



%% Correlate data to get test accuracy and make figures (2 point)

% Calculate accuracy by correlating predicted and actual angles for each
% finger separately. Hint: You will want to use zohinterp to ensure both 
% vectors are the same length.

%%
%testing prediction using linear filter

%calculating R matrix for test data
s1_window_feats_test = getWindowedFeats(s1_test_ecog_cleaned, fs_hz, win_length_ms, win_overlap_ms);
s1_R_test = create_R_matrix(s1_window_feats_test, n_wind);

disp('s1 done')

s2_window_feats_test = getWindowedFeats(s2_test_ecog_cleaned, fs_hz, win_length_ms, win_overlap_ms);
s2_R_test = create_R_matrix(s2_window_feats_test, n_wind);
disp('s2 done')

s3_window_feats_test = getWindowedFeats(s3_test_ecog_cleaned, fs_hz, win_length_ms, win_overlap_ms);
s3_R_test = create_R_matrix(s3_window_feats_test, n_wind);
disp('s3 done')

%%
%predicting y for test
%calculating training error
s1_test_pred_y = s1_R_test*s1_f;
s2_test_pred_y = s2_R_test*s2_f;
s3_test_pred_y = s3_R_test*s3_f;

%upsampling testing pred y to 1000 hz signal
s1_test_pred_y_upsamp = zoh_upsample(s1_test_pred_y,50);
s2_test_pred_y_upsamp = zoh_upsample(s2_test_pred_y,50);
s3_test_pred_y_upsamp = zoh_upsample(s3_test_pred_y,50);

%calculating testing set correlation coeff
s1_test_rho = corr(s1_test_pred_y_upsamp, s1_test_dg);
s1_test_rho = diag(s1_test_rho)'

s2_test_rho = corr(s2_test_pred_y_upsamp, s2_test_dg);
s2_test_rho = diag(s2_test_rho)'

s3_test_rho = corr(s3_test_pred_y_upsamp, s3_test_dg);
s3_test_rho = diag(s3_test_rho)'



%%
%Fucntion to upsample from windows to 1 kHz
function [x_upsampled] = zoh_upsample(x,factor)
    x_sz = size(x,1);
    x_up_sz = (x_sz)*factor;
    
    x_upsampled = zeros(x_up_sz, size(x,2));
    
    for i  = 1:x_sz
        win_start_indx = round(1 + ((i-1)*factor));
        win_end_indx = round(i*factor);
        
        x_upsampled(win_start_indx:win_end_indx,:) = x(i,:).*ones(factor,1);
    end
    
    x_upsampled = [x_upsampled;x_upsampled(end-factor+1 : end,:)];
    
end

function [feature_matrix] = avg_features(features, N_wind)

    num_win = size(features,1);
    num_features = size(features,2);

    %duplicationg the first 2 time window rows to calculate M-1 window feature
    %values for the first 2 time windows
    features_append = [features(1:N_wind-1, :);features];
    
    feature_matrix = zeros(num_win, num_features);
    %Averaging features from M, M-1 and M-2 windows
    
    for i = 1:num_win
        feature_matrix(i,:) = mean(features_append(i:i+N_wind-1,:));
    end

end

