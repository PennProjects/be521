function [features] = get_features(clean_data,fs)
    %
    % get_features.m
    %
    % Instructions: Write a function to calculate features.
    %               Please create 4 OR MORE different features for each channel.
    %               Some of these features can be of the same type (for example, 
    %               power in different frequency bands, etc) but you should
    %               have at least 2 different types of features as well
    %               (Such as frequency dependent, signal morphology, etc.)
    %               Feel free to use features you have seen before in this
    %               class, features that have been used in the literature
    %               for similar problems, or design your own!
    %
    % Input:    clean_data: (samples x channels)
    %           fs:         sampling frequency
    %
    % Output:   features:   (1 x (channels*features))
    % 
%% Your code here (8 points)

%feature functions
%line length
LLFn = @(x) sum(abs(diff(x)));
%Area
AreaFn = @(x) sum(abs(x));
%energy
EnergyFn = @(x) sum(x.^2);
%running average called lmp in kubanek2009
lmpFn = @(x) mean(x);

num_chan = size(clean_data,2);
num_feats = 4;

feature_matix = zeros(num_chan,num_feats)
for i = 1:num_chan
    line_length
end


end

