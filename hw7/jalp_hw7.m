%%
% <latex> 
% \title{BE 521: Homework 7 \\{\normalsize p300 Speller} \\{\normalsize Spring 2021}}
% \author{34 points}
% \date{Due: 3/23/2021 10 PM}
% \maketitle \textbf{Objective:} Spell letters using neurosignals
% </latex>

%% 
% <latex> 
% \begin{center} \author{Jal Mahendra Panchal}
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
%  \includegraphics[width=\textwidth]{scalpEEGChans.png}
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
%parsing eeg data for target and non-target trials
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

%%
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
% Of the two channels, it looks like channel 11, Cz has a much higher ratio of
% Target/Non-Target mean value as compared to the channel 42, T8. This
% would make it easier to detect a peak from a target stimulation in the Cz
% channel data. $\\$
% From the graphs, in the Cz channel, we see the highest peak is at about 450ms
% which could be the best point to detect the target. While on the graph
% for T8 channel the peak is at about 480 ms which would be the best point
% to detect the target signal.


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
topoplotEEG(diff_at300_uV,'eloc64.txt','gridscale', 150, 'colormap', cmap, 'electrodes', 'numbers')
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
% between the target and non-target response at those electrode locations.
% The blue color in the other hand shows areas where the non-target stimuli have
% a greater response as compared to the target stimuli. Or it could also be
% locations where the target response as more negative as compared to the
% non-target response. $\\$
% Here we see that the Cz electrode has the highest difference while the T8
% electrode has a very small difference between the target and non-target response
% at 300ms. This is consistent with the plots in 1.1and 1.2 where we
% observed a much higher ratio  and difference between target and
% non-target signal
% in Cz as compared to T8 electrode.
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
% By using only one channel we can simplify the hardware and usability of
% the whole system as then only 2 electrodes (Cz and ground) will be needed for the
% prediction. This is help optimize data storage, power and computational needs.
% $\\$
% One big disadvantage of using only one electrode is it does not help adapt to
% people who have disabilities in the brain or have impairments/damage.
% We would then not know if the region detected by the Cz electrode is the
% the appropriate one for best p300 signal. If a person has brain damage in
% that region, the device will not be usable.
% For some people due to neuroplasticity, their area of high
% activity might be different than the one detected by Cz. In both such
% cases it would make more sense to have the complete skull covered instead
% of a single or only a few electrodes.


%% 
% <latex> 
%  \item One simple way of identifying a P300 in a single trial (which we'll call the \emph{p300 score}) is to take the mean EEG from 250 to 450 ms and then subtract from it the mean EEG from 600 to 800 ms. What is the 	\emph{p300 score} for epoch (letter) 10, iteration 11 at electrode Cz? (3 pts)
% </latex>

%%
% $\textbf{Answer 2.2} \\$

%%
%Calculating pscore for Cz channel
%250ms to 450 ms is at index 61 to 108
%600 to 800 ms is from index 145-192

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
%p300\ score for epoch 10 and iterarion 11 is 
pscore_c11(10,11)

%%
% The $pscore$ for epoch 10 and iteration 11 at electrode Cz is 0.8243
% $\muV$


%% 
% <latex> 
%  \item Plot the \emph{p300 scores} for each row/column in epoch 27 at electrode Cz. (3 pts)
% </latex>

%%
% $\textbf{Answer 2.3} \\$

%%
% The spread of the p300 scores for each row/column can be shown as
% individual boxplots. 

%%
%first we must find the indexes of the 15 trials for each row and column

epoch27_stim_rowcol = zeros(1,180);
for i = 1:180
    epoch27_stim_rowcol(i) = str2double(Stim(26*180+i).description);
end

%separate the indexes for each row and col number creating a 12x15 index
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
        epoch_i_stim_rowcol(j) = str2double(Stim((i-1)*180+j).description);
        
    end

    %separate the indexes for each row and col number creating a 12x15 index
    %table
    epoch_i_idx_separated = zeros(12,15);
    pscore_temp_ = zeros(12,15);
    for k = 1:12
        epoch_i_idx_separated(k, :) = find(epoch_i_stim_rowcol==k);
        pscore_temp_(k,:) = pscore_c11(i, epoch_i_idx_separated(k, :));
    end
    
    %calculating the average p300 score across all 15 trails for each
    %row/col
    temp_ = mean(pscore_temp_,2);
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
    
    pscore_c11_allepoch(i,:);
    %parsing row and column index
    %finding highest row index
    temp1_  = find(temp_idx>6);
    row_idx = temp_idx(temp1_(1))-6;
    
    %finding highest column index
    temp2_  = find(temp_idx<7);
    col_idx = temp_idx(temp2_(1));
   
    pred_letter_c11 = letter_matrix(row_idx,col_idx);
    target_letter= TargetLetter(i).description;
    
    disp(['Predicted Letter : ', pred_letter_c11, '   Target Letter :', target_letter])
   % disp(['Predicted Letter : ', pred_letter_c11, '   Target Letter :', target_letter, '    R : ', num2str(row_idx), '    C : ', num2str(col_idx) ])
    
    c11_p300_accuracy(i) = pred_letter_c11==TargetLetter(i).description;
end


prediction_accuracy = sum(c11_p300_accuracy)/85*100

%%
% The prediction accuracy by using the highest p300 values for rows and
% columns trials and finding the intersection of the 2 is 27.06%.

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
% To improve the prediction from section 2, we will include 9 electrodes in
% the analysis. Cz and the 8 electrodes surrounding it.  We will then
% calculate a average p300 score from the 9 electrodes. We will also extend
% the timing range for p300 calculation to 250-500 ms.
%%
%Extending the timimg for p300 to 250-500ms
%Calculating average p300score for 9 channels 
%250ms to 500 ms is at index 61 to 120
%600 to 800 ms is from index 145-192

%channels 3,4,5,10,11,12,17,18,19
%the chosen channels are the 8 channels around channel 11,Cz
electrode_arr = [3,4,5,10,11,12,17,18,19];
%we'll now caculate the average p300 score for the 9 electrodes

p300score_nineavg = zeros(85,180);

for i = 1:85
    for j = 1:180
        
        %separatingindex for each trial in each epoch
        stim_idx = (i-1)*180+j;
        %finding the index fro the datafrom time stamps
        st_idx = round(((Stim(stim_idx).start)/1e6)*sampling_frequency_hz)+1;
        sp_idx = round(((Stim(stim_idx).stop)/1e6)*sampling_frequency_hz);
        
        p300_temp_ = zeros(9,1);
        for e = 1:9
            temp_ = data_uV(electrode_arr(e),st_idx:sp_idx);
            
            %p300 score = val(250-500)ms-val(600to800)
            p300_temp_(e) = mean(temp_(61:120)-mean(temp_(145:192)));
        end
       
        %calculating mean p300 value
        mean_temp_ = mean(p300_temp_);
        
        %p300 score = val(250-500)ms-val(600to800)
        p300score_nineavg(i,j) = mean_temp_;
    end
end



%%
%Calculating the mean p300 score for 12 rows and columns
%Calculating the mean p300 score for all epochs
p300score_stimavg = zeros(85,12);

for i = 1:85
    epoch_i_stim_rowcol = zeros(1,180);
    for j = 1:180
        epoch_i_stim_rowcol(j) = str2double(Stim((i-1)*180+j).description);
        
    end

    %separate the indexes for each row and col number creating a 12x15 index
    %table
    epoch_i_idx_separated = zeros(12,15);
    pscore_temp_ = zeros(12,15);
    for k = 1:12
        epoch_i_idx_separated(k, :) = find(epoch_i_stim_rowcol==k);
        pscore_temp_(k,:) = p300score_nineavg(i, epoch_i_idx_separated(k, :));
    end
    
    %calculating the average p300 score across all 15 trails for each
    %row/col
    temp_ = mean(pscore_temp_,2);
    p300score_stimavg(i,:)= temp_';
end

%%
% Now we have the average p300 for each of the 12 stim row/cols for each
% epoch. We will train a classifier for the rows and column index
% separately. To do this, for each epoch, we will create a 2 sets of
% features, the p300 score for a row and a p300 score for a column. We will
% then create separate SVM models to predict the row and column with valid p300 scores. 
% For each epoch we will have 6 data points from each row/column to train
% the model.

%%
%separating row and columns data
p300_rows = p300score_stimavg(:,7:12);
p300_cols = p300score_stimavg(:,1:6);

%separating training and testing data
p300_rows_train = p300_rows(1:50,:);
p300_rows_test = p300_rows(51:85,:);

p300_cols_train = p300_cols(1:50,:);
p300_cols_test = p300_cols(51:85,:);

%converting into a column matrix
p300_rows_train_vec = reshape(p300_rows_train,[],1);
p300_rows_test_vec = reshape(p300_rows_test, [],1);

p300_cols_train_vec = reshape(p300_cols_train, [],1);
p300_cols_test_vec = reshape(p300_cols_test, [],1);

%%normalizing the data
p300_rows_train_norm = normalize(p300_rows_train_vec);

p300_cols_train_norm = normalize(p300_cols_train_vec);

%%
%creating the labels for training and validation
all_labels = zeros(85,12);

for i = 1:85
    target_letter = TargetLetter(i).description;
    tar_row = findRow(target_letter);
    tar_col = findCol(target_letter);
    all_labels(i,tar_col) = 1;
    all_labels(i,tar_row) = 1;
end

row_labels_train = reshape(all_labels(1:50,7:12), [],1);
col_labels_train = reshape(all_labels(1:50,1:6), [],1);

%%
%using a SVM classifier
%for rows
X = p300_rows_train_norm;
Y = row_labels_train; 
svmodel_rows = fitcsvm(X,Y,  'KernelFunction','rbf')

%Calculating training error by predicting the training set
row_pred_train_svm = predict(svmodel_rows, X);

train_error_svm_rows = size(find(row_pred_train_svm~=Y),1)/size(Y,1)

%%
%SVM classifier for columns
X = p300_cols_train_norm;
Y = col_labels_train; 

svmodel_cols = fitcsvm(X,Y,  'KernelFunction','rbf')

%Calculating training error by predicting the training set
cols_pred_train_svm = predict(svmodel_cols, X);

train_error_svm_cols = size(find(cols_pred_train_svm~=Y),1)/size(Y,1)


%%
% To complete the prediction, we will check the classification of the model
% for the set of 6 rows and cols. If the model returns a 1 for a given row/column, 
% it means that is identified as a valid p300 score and that
% row/col is selected. If No row/col is classified as valid, we'll choose the
% one with the highest p300 value. If more than one row/cal are classified as valid,
% we'll choose the one with the highest p300 value.

%%
%testing/validating the models

testing_accuracy_svm = zeros(1,35); 

%Finding the predicted letter for each of the 35 testing epochs
for i = 1:35
    
    %predicting row_index
    %pre-processing test data
    rows_norm = normalize(p300_rows_test(i,:));
    test_pred_row_val = predict(svmodel_rows,rows_norm');
    
    %if no valid p300 score, select highest
    if sum(test_pred_row_val) ==0
        [~,pred_row] = max(rows_norm);
    
    %if multiple valid p300 score, select highest
    elseif sum(test_pred_row_val) > 1
        temp_ = max(rows_norm(test_pred_row_val==1));
        pred_row  = find(rows_norm==temp_);
    else
        [pred_row] = find(test_pred_row_val==1);
    end
    
    
    %Predicting column index
    %preprocessing test data
    cols_norm = normalize(p300_cols_test(i,:));
    test_pred_col_val = predict(svmodel_cols, cols_norm');
    
    %if no valid p300 score, select highest
    if sum(test_pred_col_val) ==0
        [~,pred_col] = max(cols_norm);
    
    %if multiple valid p300 score, select highest
    elseif sum(test_pred_col_val) > 1
        temp_ = max(cols_norm(test_pred_col_val==1));
        pred_col  = find(cols_norm==temp_);
    else
        pred_col = find(test_pred_col_val==1);
    end

    
    pred_letter_all = letter_matrix(pred_row,pred_col);
    target_letter = TargetLetter(50+i).description;
    disp(['Predicted Letter : ', pred_letter_all, '   Target Letter :', target_letter])
    
    testing_accuracy_svm(i) = pred_letter_all ==TargetLetter(50+i).description;
     
end

testing_accuracy = sum(testing_accuracy_svm)/35*100

%%
% The optimal accuracy I was able to obtain was 25.7 % for the training
% set. This was done using separate SVM models to classify a row/column as one with
% a valid p300 score and hence being chosen to predict the letter.

%% 
% <latex> 
%  \item Describe your algorithm in detail. Also describe what you tried that didn't work. (6 pts)
% </latex>

%%
% $\textbf{Answer 3.2} \\$

%%
% <latex>
% The steps followed in the algorithm above :
% \begin{enumerate}
% \item To begin with, to improve the prediction from section 2, I included 9 electrodes in
% the analysis. Cz and the 8 electrodes surrounding it.
% \item I begin by parsing the signals from the 9 electrodes into
% individual trails.
% \item For each trial, I calculate the p300 value for all 9 electrodes 
% using mean(250-500ms)-mean(600-800ms) and
% then calculate a mean value of the p300 from the 9 electrodes
% \item I then calculate the average p300 for the 15 trials of each of the
% 12 rows/columns giving a 85x12 matrix of mean p300 values.
% \item Then I separate the p300 values for the row and column triggers.
% \item The 85x6 matrices for the rows and columns are then split into the
% 50 training and the the 35 testing set. 
% \item The training set for rows and column is converted from a 50x6
% matrix to a 300x1 column vector so that each row/column can be classified
% as a valid p300 or not.
% \item I then normalize the 300x1 training vector for rows and columns
% separately to prepare it for classification.
% \item This 300x1 vector of p300 values for rows is used to train a SVM
% model with Gaussian kernel. The same is done for a model using the column
% values of p300. the classification on both cases is binary with a 1 given
% to the valid p300 row/column and a 0 to a an invalid p300 score for a row/column.
% \item The model is then used to predict data from a new trial set by predicting a
% row or column value. 
% \item For testing I followed the same averaging
% process as training using the 9 electrodes, finding the averaging p300 value and
% averaging the 15 trials for each row/column.
% \item the data is then split into rows and columns, giving 6 inputs
% values to each model to predict a valid p300 score.
% \item If the model classiifies more than one row/column as a valid p300
% then the one with the higher p300 is choosen. If the model does not
% classify any row/column as a valid p300 then the row/column value with the highest
% p300 for the trial is chosen. 
% \item We then finally get a chosen row and column value. The
% intersection of these two is  the predicted letter. This is then compared
% to the Target letter to verify the model accuracy.
% \end{enumerate}
% \\ \\
% Other methods tried : 
% \begin{enumerate}
% \item I calculated the p300 score using the ratio of values
% from 250-500 ms to those from 600-800 ms. The ratio was created a lot of
% anomalies due to the decimal and negative values in the data and so was
% discarded.
% \item I tried the model using only Cz and using more electrodes. Having
% the 9 electrodes gave the most stable results for accuracy
% \item I also tried averaging the signal across the electrodes and then
% calculating the p300 values. As compared to calculating p300 and then
% averaging the value for the 9 electrodes.
% \item I also tried using fitcecoc to train a multi-class classifier to
% classify the p300 values for rows and columns in 6 classes. This gave
% worse results than the current algorithm. Possibly due to a small sample
% size for 6 class classifier.
% \end{enumerate}
% </latex>

%% 
% <latex> 
% \end{enumerate}
% \end{document}
% </latex>
