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
%parsing eeg data fortarget and nontarget trials
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
%parsing eeg data fortarget and nontarget trials
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
% <latex> 
%  \item Compute the mean difference between the target and non-target stimuli for each channel at timepoint 300 ms averaged across all row/column flashes. Visualize these values using the \verb|topoplotEEG| function. Include a colorbar. (3 pts)
% </latex> 

%%
% $\textbf{Answer 1.4} \\$

%% 
% <latex> 
%  \item How do the red and blue parts of this plot correspond to the plots from above? (2 pts)
% </latex> 

%%
% $\textbf{Answer 1.5} \\$

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
% <latex> 
%  \item One simple way of identifying a P300 in a single trial (which we'll call the \emph{p300 score}) is to take the mean EEG from 250 to 450 ms and then subtract from it the mean EEG from 600 to 800 ms. What is the 	\emph{p300 score} for epoch (letter) 10, iteration 11 at electrode Cz? (3 pts)
% </latex>

%%
% $\textbf{Answer 2.2} \\$
%% 
% <latex> 
%  \item Plot the \emph{p300 scores} for each row/column in epoch 27 at electrode Cz. (3 pts)
% </latex>

%%
% $\textbf{Answer 2.3} \\$

%% 
% <latex> 
%  \item Based on your previous answer for epoch 27, what letter do you predict the person saw? Is this prediction correct? (2 pts)
% </latex>

%%
% $\textbf{Answer 2.4} \\$

%% 
% <latex> 
%  \item Using this \emph{p300 score}, predict (and print out) the letter viewed at every epoch. What was you prediction accuracy? (2 pts)
% </latex>

%%
% $\textbf{Answer 2.5} \\$

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
