% function [all_feats]=getWindowedFeats(raw_data, fs, window_length, window_overlap)
    %
    % getWindowedFeats.m
    %
    % Instructions: Write a function which processes data through the steps
    %               of filtering, feature calculation, creation of R matrix
    %               and returns features.
    %
    %               Points will be awarded for completing each step
    %               appropriately (note that if one of the functions you call
    %               within this script returns a bad output you won't be double
    %               penalized)
    %
    %               Note that you will need to run the filter_data and
    %               get_features functions within this script. We also 
    %               recommend applying the create_R_matrix function here
    %               too.
    %
    % Inputs:   raw_data:       The raw data for each patients as (samples
    %                            x channels)
    %           fs:             The raw sampling frequency
    %           window_length:  The length of window
    %           window_overlap: The overlap in window
    %
    % Output:   all_feats:      All calculated features
    %
%% Your code here (3 points)

%Here we will work on one subject at a time 
s1_train_ecog_cleaned = train_ecog{1}(1:225000,:);
raw_data = s1_train_ecog_cleaned;
fs = 1000;
window_length = 100;
window_overlap = 50;

%We will first normalise data from each channel
norm_data = normalize(raw_data);

%filter data
filtered_data = filter_data(norm_data);

%common normal reference montage
%at everytime point, subtracting mean of othe channels from value of a given
%channel
reref_data = filtered_data - mean(filtered_data,2);

%separation data into sliding windows
%each window will have (samples_in_window x channels) amount of data
xLen = size(reref_data,1);
NumWins =floor((xLen-(window_overlap))/(window_length - window_overlap));

window_disp = window_length-window_overlap;

for i = 1:5
    win_start_indx = round(1 + ((i-1)*window_disp));
    win_end_indx = round(win_start_indx +window_length-1);
    
    %extractiong data for one window as (samples_in_window x channels) amount of data
    window_data = filtered_data(win_start_indx : win_end_indx, :);
    
    %extracting features for each window by calling get_features
    wind_features = get_features(window_data, fs);
end    


% Finally, return feature matrix

% end