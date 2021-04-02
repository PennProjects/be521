function clean_data = filter_data(raw_eeg)
    %
    % filter_data.m
    %
    % Instructions: Write a filter function to clean underlying data.
    %               The filter type and parameters are up to you.
    %               Points will be awarded for reasonable filter type,
    %               parameters, and correct application. Please note there 
    %               are many acceptable answers, but make sure you aren't 
    %               throwing out crucial data or adversely distorting the 
    %               underlying data!
    %
    % Input:    raw_eeg (samples x channels)
    %
    % Output:   clean_data (samples x channels)
    % 
%% Your code here (2 points) 

%designing a bandpass filter
% fl = 0.15 Hz, fh = 200 hz
pass_band = [0.15,200];
fs_hz = 1000;
[~,d] = bandpass(raw_eeg, pass_band,fs_hz);

clean_data = filter(d,raw_eeg);
    
end