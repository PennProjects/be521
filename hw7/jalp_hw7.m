%%
% <latex> 
% \title{BE 521: Homework 7 \\{\normalsize p300 Speller} \\{\normalsize Spring 2021}}
% \author{34 points}
% \date{Due: 3/23/2021 10 PM}
% \maketitle \textbf{Objective:} Spell letters using neurosignals
% </latex>

%% 
% <latex> 
% \begin{center} \author{NAME HERE \\
%   \normalsize Collaborators: COLLABORATORS HERE \\}
% \end{center} 
% </latex>

%% 
% <latex> 
% \subsection*{P300 Speller}
% In this homework, you will work with data from a P300-based brain computer interface called BCI2000 (Schalk et al. 2004) that allows people to spell words by focusing their attention on a particular letter displayed on the screen. In each trial the user focused on a letter, and when that letter's row or column is flashed, the user's brain elicits a P300 evoked response. By analyzing whether a P300 signal was produced, across the flashes of many different rows or columns, the computer can determine the letter that the person is focusing on.
% </latex>

%% 
% <latex> 
% Figure 1 shows the letter matrix from one trial of this task.
% \begin{figure}
%  \centering
%  \includegraphics[width=0.3\textwidth]{letterMat}
%  \caption{The letter matrix for the P300 speller with the third row illuminated.
%           If the user were focusing on any of the letters in the third row (M, N, O, P, Q, or R),
%           their brain would emit a P300 response. Otherwise it would not.}
% \end{figure}
% </latex>

%% 
% <latex> 
% \subsection*{Data Organization}
% The data for this homework is stored in \verb|I521_A0008_D001| on the IEEG Portal.
% The EEG in this dataset were recorded during 85 intended letter spellings. For each letter spelling, 12 row/columns were flashed 15 times in random order ($12 \times 15 = 180$ iterations). The EEG was recorded with a sampling rate of 240 Hz on a 64-channel scalp EEG.\\
% </latex>

%% 
% <latex> 
% The annotations for this dataset are organized in two layers as follows:
% \begin{itemize}
%     \item \verb|TargetLetter| annotation layer indicates the target
%     letter (annotation.description) on which the user was focusing
%     during the recorded EEG segment
%     (annotation.start/annotation.stop). This layer is also provided
%     as TargetLetterAnnots.mat.
%     \item \verb|Stim| annotation layer indicates the row/column that
%     is being flashed (annotation.description) and whether the target
%     letter is contained in that flash (annotation.type). The
%     recorded EEG during that flash is
%     (annotation.start/annotation.stop). Note that this annotation
%     layer is provided as StimAnnots.mat. It is NOT on the portal.
% \end{itemize}
% Hints: There are many annotations in this dataset and getting them all may take ~5-10 minutes. Once you retrieve the annotations once, save them for faster loading in the future. Also, use $\verb|{ }|$ to gather variables across structs for easier manipulation (e.g. $\verb|strcmp({annotations.type},'1')|$) \\
% </latex>

%% 
% <latex> 
% \begin{figure}
%  \centering
%  \includegraphics[width=0.3\textwidth]{letterMatInds}
%  \caption{The row/column indices of the letter matrix, as encoded in the \textbf{Stim} annotation layer (annotation.description) matrix.}
% \end{figure}
% </latex>

%% 
% <latex> 
% \subsection*{Topographic EEG Maps}
% You can make topographic plots using the provided \verb|topoplotEEG| function. This function needs an ``electrode file.'' and can be called like
% \begin{lstlisting}
%  topoplotEEG(data,'eloc64.txt','gridscale',150)
% \end{lstlisting}
% where \verb|data| is the value to plot for each channel. This function plots the electrodes according to the map in Figure 3.
% \begin{figure}
%  \centering
%  \includegraphics[width=\textwidth]{scalpEEGChans}
%  \caption{The scalp EEG 64-channel layout.}
% \end{figure}
% </latex> 

%% 
% <latex> 
% \pagebreak
% \section{Exploring the data}
% In this section you will explore some basic properties of the data in \verb|I521_A0008_D001|.
% </latex> 

%% 
% <latex> 
% \begin{enumerate}
% </latex> 

%% 
% <latex> 
%  \item For channel 11 (Cz), plot the mean EEG for the target and non-target stimuli separately, (i.e. rows/columns including and not-including the desired character, respectively), on the same set of axes. Label your x-axis in milliseconds. (3 pts)
% </latex> 

%%
% $\textbf{Answer 1.1} \\$

%% 
%fetching data form the data base I521_A0008_D001
addpath(genpath('/Users/jalpanchal/git/be521'));

session = IEEGSession('I521_A0008_D001', 'jalpanchal', 'jal_ieeglogin.bin');
sampling_frequency_hz = session.data.sampleRate;
duration_in_sec = session.data(1).rawChannels(1).get_tsdetails.getDuration/1e6;

data_uV = [];
for i = 1:64
    data_uV(i,:) = session.data.getvalues(0, 15300 * 1e6, i);
end

%%
%load annotations 
load('TargetLetterAnnots.mat')
load('StimAnnots.mat')

%%
%making a function to find the row/column index of a letter
letter_matrix = ['A', 'B', 'C', 'D', 'E', 'F';
                 'G', 'H', 'I', 'J', 'K', 'L';
                 'M', 'N', 'O', 'P', 'Q', 'R';
                 'S', 'T', 'U', 'V', 'W', 'X';
                 'Y', 'Z', '1', '2', '3', '4';
                 '5', '6', '7', '8', '9', '_';];
             
findCol = @(x) floor(find(letter_matrix==x)/6)+1;
findRow = @(x) (mod(find(letter_matrix==x),6)+6);

%%
%separating the trial numbers for target and non-target stimuli
target_stim_index = zeros(85,30);
nontarget_stim_index = zeros(85,150);

for i = 1:85
    target_letter = TargetLetter(i).description;
    target_row = findRow(target_letter);
    target_col = findCol(target_letter);
    temp_  = [];
    temp1_ = [];
    for j = 1:180
        stim_idx = (i-1)*180+j;
        stm_rowcol = str2double(Stim(stim_idx).description);
        if(stm_rowcol == target_row || stm_rowcol == target_col)
            temp_ = [temp_,stim_idx];
        else
            temp1_ = [temp1_,stim_idx];
        end
    end
    target_stim_index(i,:) = temp_;
    nontarget_stim_index(i,:) = temp1_;
end


%%
%parsing eeg data for target and nontarget trials
%Here each row represents data for 1 epoch

%each row has 240 points. This is the average of the 30 target trials 
e11_target_uV = zeros(85,240);

for i = 1:85
    temp_ = [];
    temp1_ = [];
    for j = 1:30
        stm_idx = target_stim_index(i,j);
        st_idx = round(((Stim(stm_idx).start)/1e6)*sampling_frequency_hz)+1;
        sp_idx = round(((Stim(stm_idx).stop)/1e6)*sampling_frequency_hz);
        temp_ = data_uV(11,st_idx:sp_idx);
        temp1_ = [temp1_; temp_];
    end
    e11_target_uV(i,:) = mean(temp1_);
end

%each row has 240 points. This is the average of the 150 non-target trials 
e11_nontarget_uV = zeros(85,240);

for i = 1:85
    temp_ = [];
    temp1_ = [];
    for j = 1:150
        stm_idx = nontarget_stim_index(i,j);
        st_idx = round(((Stim(stm_idx).start)/1e6)*sampling_frequency_hz)+1;
        sp_idx = round(((Stim(stm_idx).stop)/1e6)*sampling_frequency_hz);
        temp_ = data_uV(11,st_idx:sp_idx);
        temp1_ = [temp1_; temp_];
    end
    e11_nontarget_uV(i,:) = mean(temp1_);
end

%%
%Calculating the average voltage 
e11_target_avg_uV = mean(e11_target_uV);
e11_nontarger_avg_uV = mean(e11_nontarget_uV);

%%
%plotting mean signals
t = 0 : 1/sampling_frequency_hz : 1-1/sampling_frequency_hz;
t_ms = t*1e3;

figure();
plot(t_ms, e11_target_avg_uV, 'Linewidth', 2);
hold on 
plot(t_ms, e11_nontarger_avg_uV, 'Linewidth', 2);
xlabel('Time(ms)')
ylabel('Signal Amplitude(\muV)')
title('Comparing target vs non-target response for Cz channel(11)')
legend('Target', 'Non-Target')


% <latex> 
%  \item Repeat the previous questions for channel 42 (T8). (1 pts)
% </latex> 

%%
% $\textbf{Answer 1.2} \\$

%%
%parsing eeg data for target and nontarget trials
%Here each row represents data for 1 epoch

%each row has 240 points. This is the average of the 30 target trials 
e42_target_uV = zeros(85,240);

for i = 1:85
    temp_ = [];
    temp1_ = [];
    for j = 1:30
        stm_idx = target_stim_index(i,j);
        st_idx = round(((Stim(stm_idx).start)/1e6)*sampling_frequency_hz)+1;
        sp_idx = round(((Stim(stm_idx).stop)/1e6)*sampling_frequency_hz);
        temp_ = data_uV(42,st_idx:sp_idx);
        temp1_ = [temp1_; temp_];
    end
    e42_target_uV(i,:) = mean(temp1_);
end

%each row has 240 points. This is the average of the 150 non-target trials 
e42_nontarget_uV = zeros(85,240);

for i = 1:85
    temp_ = [];
    temp1_ = [];
    for j = 1:150
        stm_idx = nontarget_stim_index(i,j);
        st_idx = round(((Stim(stm_idx).start)/1e6)*sampling_frequency_hz)+1;
        sp_idx = round(((Stim(stm_idx).stop)/1e6)*sampling_frequency_hz);
        temp_ = data_uV(42,st_idx:sp_idx);
        temp1_ = [temp1_; temp_];
    end
    e42_nontarget_uV(i,:) = mean(temp1_);
end

%%
%Calculating the average voltage 
e42_target_avg_uV = mean(e42_target_uV);
e42_nontarger_avg_uV = mean(e42_nontarget_uV);

%%
%plotting mean signals
t = 0 : 1/sampling_frequency_hz : 1-1/sampling_frequency_hz;
t_ms = t*1e3;

figure();
plot(t_ms, e42_target_avg_uV, 'Linewidth', 2);
hold on 
plot(t_ms, e42_nontarger_avg_uV, 'Linewidth', 2);
xlabel('Time(ms)')
ylabel('Signal Amplitude(\muV)')
title('Comparing target vs non-target response for T8 channel(42)')
legend('Target', 'Non-Target')

%% 
% <latex> 
%  \item Which of the two previous channels looks best for distinguishing between target and non-target stimuli? Which time points look best? Explain in a few sentences. (2 pts)
% </latex> 

%%
% $\textbf{Answer 1.3} \\$
%%
e11_t_m = mean(e11_target_avg_uV);
e11_nt_m = mean(e11_nontarger_avg_uV);
snr_ratio_e11 = e11_t_m/e11_nt_m

e42_t_m = mean(e42_target_avg_uV);
e42_nt_m = mean(e42_nontarger_avg_uV);
snr_ratio_e42 = e42_t_m/e42_nt_m

%%
% Of the channels, it looks like channel 11, Cz has a much higher ratio of
% Target/Non-Terget mean value as compared to the channel 42, T8. This
% would make it easier to detect a peak from a target stimulation in the Cz
% channel data. $\\$
% From the graphs, in the Cz channel, we havethe highest peak at abour 450
% ms which could be the best point to detect the target. While on the graph
% for T8 channel the peak is at about 480 ms which would be the best point
% to gete the target signal.


%% 
% <latex> 
%  \item Compute the mean difference between the target and non-target stimuli for each channel at timepoint 300 ms averaged across all row/column flashes. Visualize these values using the \verb|topoplotEEG| function. Include a colorbar. (3 pts)
% </latex> 

%%
% $\textbf{Answer 1.4} \\$

%%
%calculating the mean response for each channel for all trials for all
%letters
%We'll calculate the mean of the 30 target trials for each letter/epoch.
%Then we will average the response acress each epoch to get a final 1s
%average response

all_target_avg_uV = zeros(64,240);

for e = 1:64
    temp2_ = zeros(85,240);
    for i = 1:85
        temp_ = [];
        temp1_ = [];
        for j = 1:30
            stm_idx = target_stim_index(i,j);
            st_idx = round(((Stim(stm_idx).start)/1e6)*sampling_frequency_hz)+1;
            sp_idx = round(((Stim(stm_idx).stop)/1e6)*sampling_frequency_hz);
            temp_ = data_uV(e,st_idx:sp_idx);
            temp1_ = [temp1_; temp_];
        end
        %each row has 240 points. This is the average of the 30 target trials 
        temp2_(i,:) = mean(temp1_);
    end
    %Calculating avg across all 85 letters/epochs
    all_target_avg_uV(e, :) = mean(temp2_);   
end

%We'll calculate the mean of the 150 non target trials for each letter/epoch.
%Then we will average the response acress each epoch to get a final 1s
%average response

all_nontarget_avg_uV = zeros(64,240);

for e=1:64
    temp2_ = zeros(85,240);
    for i = 1:85
        temp_ = [];
        temp1_ = [];
        for j = 1:150
            stm_idx = nontarget_stim_index(i,j);
            st_idx = round(((Stim(stm_idx).start)/1e6)*sampling_frequency_hz)+1;
            sp_idx = round(((Stim(stm_idx).stop)/1e6)*sampling_frequency_hz);
            temp_ = data_uV(e,st_idx:sp_idx);
            temp1_ = [temp1_; temp_];
        end
        %each row has 240 points. This is the average of the 150 non target trials 
        temp2_(i,:) = mean(temp1_);
    end
    %Calculating avg across all 85 letters/epochs
    all_nontarget_avg_uV(e, :) = mean(temp2_);   
end


%%
%Calculating the difference betweeen target and non target signals at
%300 ms. The 73rd data point contain data from 300-304.167 ms
diff_at300_uV = (all_target_avg_uV(:,73))-(all_nontarget_avg_uV(:,73));

%%
%Plotting target-nontarget values at 300ms
figure();
cmap = colormap(jet);
topoplotEEG(diff_at300_uV,'eloc64.txt','gridscale', 150, 'colormap', cmap, 'electrodes', 'labels')
h = colorbar;
title('Difference in uV between target and non-taregt stimuli at 300 ms')

%% 
% <latex> 
%  \item How do the red and blue parts of this plot correspond to the plots from above? (2 pts)
% </latex> 

%%
% $\textbf{Answer 1.5} \\$

%%
% In the color bar chosen, red color signifies the largest difference
% between the target and non-tartet response at those electrode locations.
% The blue color in the other hand shows areas where the non-target stimuli have
% a greater response as compared to the target stimuli. Or it could also be
% locations where the target response as more negative as compared to the
% non-target response. $\\$
% Here we see that the Cz electrode has the highest difference while the T8
% electrode has a very small difference between the target and non-target respose
% at 300ms. This is consistent with the plots in 1.1and 1.2 where we
% observed a might higher ratio and difference in Cz as compared to T8
% electrode.
%% 
% <latex> 
% \end{enumerate}
% \section{Using Individual P300s in Prediction}
% Hopefully the Question 1.4 convinced you that the Cz channel is a reasonably good channel to use in separating target from non-target stimuli in the P300. For the rest of the homework, you will work exclusively with this channel. 
% \begin{enumerate}
%  \item Explain a potential advantage to using just one channel other than the obvious speed of calculation advantage. Explain one disadvantage. (3 pts)
% </latex> 

%%
% $\textbf{Answer 2.1} \\$

%%
% By just using one channel 

%% 
% <latex> 
%  \item One simple way of identifying a P300 in a single trial (which we'll call the \emph{p300 score}) is to take the mean EEG from 250 to 450 ms and then subtract from it the mean EEG from 600 to 800 ms. What is the 	\emph{p300 score} for epoch (letter) 10, iteration 11 at electrode Cz? (3 pts)
% </latex>

%%
% $\textbf{Answer 2.2} \\$

%%
%Calculating pscore for Cz channel
% 250ms to 450 ms is at index 61 to 108
% 600 to 800 ms is from index 145-192

pscore_c11 = zeros(85,180);

for i = 1:85
    for j = 1:180
        %separatingindex for each trial in each epoch
        stim_idx = (i-1)*180+j;
        %finding the index fro the datafrom time stamps
        st_idx = round(((Stim(stim_idx).start)/1e6)*sampling_frequency_hz)+1;
        sp_idx = round(((Stim(stim_idx).stop)/1e6)*sampling_frequency_hz);
        temp_ = data_uV(11,st_idx:sp_idx);
        
        %pscore = val(250-450)ms-val(600to800)
        pscore_c11(i,j) = mean(temp_(61:108))-mean(temp_(145:192));
    end
end

%%
%pscore for epoch 10 and iterarion 11 is 
pscore_c11(10,11)

%%
% The $pscore$ for epoch 10 and iteration 11 at electrode Cz is 0.8243


%% 
% <latex> 
%  \item Plot the \emph{p300 scores} for each row/column in epoch 27 at electrode Cz. (3 pts)
% </latex>

%%
% $\textbf{Answer 2.3} \\$

%%
%first we must find the indexes of the 15 trials for each row and column

epoch27_stim_rowcol = zeros(1,180);
for i = 1:180
    epoch27_stim_rowcol(i) = str2double(Stim(26*180+i).description);
end

%separate the indexes for ech row and col number creating a 12x15 index
%table
epoch_27_idx_separated = zeros(12,15);
pscore_c11_separated = zeros(12,15);
for i = 1:12
    epoch_27_idx_separated(i, :) = find(epoch27_stim_rowcol==i);
    pscore_c11_separated(i,:) = pscore_c11(27, epoch_27_idx_separated(i, :));
end

%%
%boxplot for all iterations of each row/column 
figure();
boxplot(pscore_c11_separated')
ylabel('p300 score (\muV)')
xlabel('Row/Column Number')
title('p300 score value across trials for each row/column at Cz, epoch 27')

%% 
% <latex> 
%  \item Based on your previous answer for epoch 27, what letter do you predict the person saw? Is this prediction correct? (2 pts)
% </latex>

%%
% $\textbf{Answer 2.4} \\$

%%
% Based the values from 2.3, we see that the Column #2 and Row #7 have the
% highest mean. When we see the intersection of the 2 values, the letter is
% B. This matches the epoch 27 in the TargetLetter annotation.
%% 
% <latex> 
%  \item Using this \emph{p300 score}, predict (and print out) the letter viewed at every epoch. What was you prediction accuracy? (2 pts)
% </latex>

%%
% $\textbf{Answer 2.5} \\$

%%
%Calculating the p300 score for all epochs
pscore_c11_allepoch = zeros(85,12);

for i = 1:85
    epoch_i_stim_rowcol = zeros(1,180);
    for j = 1:180
        epoch_i_stim_rowcol(i) = str2double(Stim((i-1)*180+j).description);
    end

    %separate the indexes for ech row and col number creating a 12x15 index
    %table
    epoch_i_idx_separated = zeros(12,15);
    pscore_c11_temp_ = zeros(12,15);
    for j = 1:12
        epoch_i_idx_separated(j, :) = find(epoch27_stim_rowcol==j);
        pscore_c11_temp_(j,:) = pscore_c11(i, epoch_i_idx_separated(j, :));
    end
    
    %calculating the average p300 score across all 15 trails for each
    %row/col
    temp_ = mean(pscore_c11_temp_,2);
    pscore_c11_allepoch(i,:)= temp_';
end


%%
%finding the predicted letter in each epoch
%We'll find the row/col with the 2 highest p300 scores
%find the intersection of the row and column to find the predicted letter

c11_p300_accuracy = zeros(1,85);
for i = 1:85
    
    %find index for row and col with 2 highest p300 values
    [~,temp_idx] = sort(pscore_c11_allepoch(i,:), 'descend');
    
    %parsing row and column index
    %finding highest row index
    temp1_  = find(temp_idx>6);
    row_idx = temp_idx(temp1_(1))-6;
    
    %finding highest column index
    temp2_  = find(temp_idx<7);
    col_idx = temp_idx(temp2_(1));
 
    pred_letter = letter_matrix(row_idx,col_idx); 
    
    c11_p300_accuracy(i) = pred_letter==TargetLetter(i).description;
end




%% 
% <latex> 
% \end{enumerate}
% \section{Automating the Learning}
% In Section 2, you used a fairly manual method for predicting the letter. Here, you will have free rein to use put any and all learning techniques to try to improve your testing accuracy. 
% \begin{enumerate}
%  \item Play around with some ideas for improving/generalizing the prediction paradigm used in the letter prediction. Use the first 50 letter epochs as the training set and the later 35 for validation. Here, you are welcome to hard-code in whatever parameters you like/determine to be optimal. What is the optimal validation accuracy you get? Note: don't worry too much about accuracy, we are more interested in your thought process. (4 pts)
% </latex>

%%
% $\textbf{Answer 3.1} \\$

%% 
% <latex> 
%  \item Describe your algorithm in detail. Also describe what you tried that didn't work. (6 pts)
% </latex>

%%
% $\textbf{Answer 3.2} \\$

%% 
% <latex> 
% \end{enumerate}
% \end{document}
% </latex>
