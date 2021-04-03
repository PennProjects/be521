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
LmpFn = @(x) mean(x);

%calculating band power for different frequency bands as specified by
%kubanek2009
%8–12 Hz, 18–24 Hz, 75–115 Hz, 125–159 Hz, 159–175 Hz
freqs = 0:1:500;

num_chan = size(clean_data,2);
num_feats = 6;

feature_matix = zeros(num_chan,num_feats);
for i = 1:num_chan
    x = clean_data(:,i);
    line_length = LLFn(x);
    area = AreaFn(x);
    energy = EnergyFn(x);
    lmp = LmpFn(x);
    
    [pxx, f] = periodogram(x,[],freqs,fs);
%     pwr_8_12 = bandpower(pxx(9:13),f(9:13),'psd');
%     pwr_18_24 = bandpower(pxx(19:25),f(19:25),'psd');
    pwr_75_115 = bandpower(pxx(76:116),f(76:116),'psd');
    pwr_125_159 = bandpower(pxx(126:160),f(126:160),'psd');
%     pwr_159_175 = bandpower(pxx(160:176),f(160:176),'psd');
    
    feature_matix(i,:) = [line_length,area, energy, lmp, pwr_75_115, pwr_125_159];
end

features = reshape(feature_matix',[],1)';
end
