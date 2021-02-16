
function [feature_values, ni, win_start, win_end] = MovingWinFeat(x, fs, winLen, winDisp, featFn)

%% Input
% x - signal as a row vector
% fs - sampling frequency in Hz
% winLen - Length of window in ms
% winDisp - Displacement of window in ms
% featFn - Feature function

%% Output
%feature_values = A vector of values of a feature from function
%                 featFn for each window
% ni = right shift of first window 
% win_start = starting index of each window
% win_end = ending index of each window

%% Definition %%

xLen = size(x,2)/fs*1e3;
% Calcualting number of windows
NumWins =floor((xLen-(winLen-winDisp))/(winDisp));
% Calculating the right shift of first window (if present)
ni = rem((xLen-(winLen-winDisp)),(winDisp));


% Trimming the portion of the signal from start which is not part of any window
x_trim = x(ni*1e-3*fs+1 : end);
x_windows = [];
win_start = [];
win_end = [];


%%Separating each window
for i = 1:NumWins
    win_start_indx = round(1 + ((i-1)*winDisp*1e-3*fs));
    win_end_indx = round(win_start_indx +winLen*1e-3*fs-1);
    window_points = x_trim(win_start_indx: win_end_indx);
    
    win_start = [win_start, win_start_indx];
    win_end = [win_end, win_end_indx];
    x_windows = [x_windows, window_points];  
end

%%Constructing a matrix with each row having elements of 1 window
x_windows = reshape(x_windows',winLen*1e-3*fs, [])';


%%Computing feature for each window
feature_values = [];
for i = 1:NumWins
    window = x_windows(i,:);
    feature_values = [feature_values, featFn(window)];
end

end
