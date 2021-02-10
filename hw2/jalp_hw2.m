%%
% <latex>
% \title{BE 521: Homework 2 Questions \\{\normalsize Modeling Neurons} \\{\normalsize Spring 2021}}
% \author{46 points}
% \date{Due: Tuesday, 2/9/2021 10:00 PM}
% \maketitle
% \textbf{Objective:} Computational modeling of neurons. \\
% We gratefully acknowledge Dr. Vijay Balasubramanian (UPenn) for many of
% the questions in this homework.\\
% </latex>

%% 
% <latex>
% \begin{center}
% \author{Jal Mahendra Panchal}
% \end{center}
% </latex>

%%
% <latex>
% \section{Basic Membrane and Equilibrium Potentials (6 pts)}
% Before undertaking this section, you may find it useful to read pg.
% 153-161 of Dayan \& Abbott's \textit{Theoretical Neuroscience} (the 
% relevant section of which, Chapter 5, is posted with the homework). 
% </latex>

%%
% <latex>
% \begin{enumerate}
%  \item Recall that the potential difference $V_T$ when a mole of ions crosses a cell membrane is defined by the universal gas constant $R = 8.31\; {\rm J/mol\cdot K}$, the temperature $T$ (in Kelvin), and Faraday's constant $F = 96,480 {\rm\ C/mol}$ \[ V_T = \frac{RT}{F} \] Calculate $V_T$ at human physiologic temperature ($37\; ^{\circ} \rm C$). (1 pt)
% </latex>

%%
% <latex>
% \rule{\textwidth}{1pt}
% \textit{Example Latex math commands that uses the align tag to make your equations
% neater. You can also input math into sentences with \$ symbol like $\pi + 1$.}
% \begin{align*}
% E = MC^2 \tag{not aligned}\\
% E = & MC^2 \tag{aligned at = by the \&}\\
% 1 = &\; \frac{2}{2}\tag{aligned at = by \&}
% \end{align*}
% \rule{\textwidth}{1pt}
% </latex>

%%
% <latex>
% \\ Answer: \\
% </latex>
%%
% Replacing the value of R = 8.134 J/ mol.K, F  = 96,480 C/mol and T = 310 K
% in the equation, we get : 
R = 8.314;
F = 96480;
T = 310;
V_t = R*T/F

%%
% <latex>
% The $V_T$ at human physiological temperatures is 0.0267 V or 26.7 mV.\\
% </latex>


%%
% <latex>
%  \item Use this value $V_T$ to calculate the Nernst equilibrium potentials 
%  (in mV) for the $\rm K^+$, $\rm Na^+$, and $\rm Cl^-$ ions, given the following 
%  cytoplasm and extracellular concentrations in the squid giant axon: 
%  $\rm K^+$ : (120, 4.5), $\rm Na^+$ : (15, 145), and $\rm Cl^-$ : (12, 120), 
%  where the first number is the cytoplasmic and the second the extracellular 
%  concentration (in mM). (2 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% To calcualte the Nernst equilibrium potential, we can use the equation,
% \begin{align*}
% E = & \frac{V_T}{z} ln(\frac{[extracellular\ concentration]}{[cytoplasmic\ concentration]})
% \end{align*}
% </latex>

K_extcell_conc_mM =4.5;
K_cyto_conc_mM = 120;
K_z = 1;

Na_extcell_conc_mM =145;
Na_cyto_conc_mM = 15;
Na_z  = 1;

Cl_extcell_conc_mM =120;
Cl_cyto_conc_mM = 12;
Cl_z = -1;

% Nernst equilibrium potential for K in mV is
E_K_mV = V_t/K_z*log(K_extcell_conc_mM/K_cyto_conc_mM)

% Nernst equilibrium potential for Na in mV is 
E_Na_mV = V_t/Na_z*log(Na_extcell_conc_mM/Na_cyto_conc_mM)

% Nernst equilibrium potential for Cl in mV is 
E_Cl_mV = V_t/Cl_z*log(Cl_extcell_conc_mM/Cl_cyto_conc_mM)

%%
% <latex>
% The Nernst equilibrium potential for $K^+$ is -87.7 mV, for $Na^+$ is 60.6
% mV and for $Cl^-$ is -61.5 mV. \\
% </latex>



%%
% <latex>
%  \item 
%   \begin{enumerate}
% 	\item Use the Goldmann equation,
% 	  \begin{equation}
% 		V_m = V_T\ln\left( \frac{\rm P_{K}\cdot[K^+]_{out} + P_{NA}\cdot[Na^+]_{out} + P_{Cl}\cdot[Cl^-]_{in}}{\rm P_{K}\cdot[K^+]_{in} + P_{NA}\cdot[Na^+]_{in} + P_{Cl}\cdot[Cl^-]_{out}} \right)
% 	  \end{equation}
% 	to calculate the resting membrane potential, $V_m$, assuming that the ratio of membrane permeabilities $\rm P_K:P_{Na}:P_{Cl}$ is $1.0:0.045:0.45$. Use the ion concentrations given above in Question 1.2. (2 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% </latex>

%%
%Using the given ratios of membrane permiabilities, we get
P_K = 1;
P_Na = 0.045;
P_Cl = 0.45;

V_m_rest = V_t*log(((P_K*K_extcell_conc_mM)+(P_Na*Na_extcell_conc_mM)+ (P_Cl*Cl_cyto_conc_mM))/((P_K*K_cyto_conc_mM)+(P_Na*Na_cyto_conc_mM)+(P_Cl*Cl_extcell_conc_mM)))

%%
% <latex>
% The resting membrane potential is $V_m$ = -63.2 mV.\\
% </latex>

%%
% <latex>
% 	\item Calculate the membrane potential at the peak action potential, assuming a permeability ratio of $1.0:11:0.45$, again using the ion concentrations given in Question 1.2. (1 pt)
%   \end{enumerate}
% </latex>

%%
% <latex>
% \\ Answer: \\
% </latex>

%%
%Using the membrane permiabilities  to find membrane potential at peak
%action potential

P_K = 1;
P_Na = 11;
P_Cl = 0.45;

V_m_actpot = V_t*log(((P_K*K_extcell_conc_mM)+(P_Na*Na_extcell_conc_mM)+(P_Cl*Cl_cyto_conc_mM))/((P_K*K_cyto_conc_mM)+(P_Na*Na_cyto_conc_mM)+(P_Cl*Cl_extcell_conc_mM)))

%%
% <latex>
% The membrane potential at peak action potential is $V_m$ = 41.5 mV.\\
% </latex>

%%
% <latex>
% 	\item The amplitudes of the multi-unit signals in HW0 and local field
% 	potentials (LFP) in HW1 had magnitudes on the order of 10 to 100
% 	microvolts. The voltage at the peak of the action potential (determined
% 	using the Goldman equation above) has a magnitude on the order of 10
% 	millivolts. Briefly explain why we see this difference in magnitude.
% 	Hint 1: Voltage is the difference in electric potential between two
% 	points. What are the two points for our voltage measurement in the
% 	multi-unit and LFP signals? What are the two points for the voltage
% 	measurement of the action potential? Hint 2: The resistance of the neuronal membrane is typically much higher than the resistance of the extracellular fluid. (2 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% Local field Potential is measured between 2 electrodes in the
% extracelular fluid in constast the action potential is measured between
% an electrode inserted into the brain cell and another electrode in the
% extracellular fluid. The resistance across the 2 electrodes for action
% potential measurement is very high due to the presence of neuronal
% membrane in between as compared to the extracellular fluid between the 2
% electrodes of during the Local field potential measurement. The
% difference in reference points and the resistnace account for the much
% highe values of action potential voltage and local field potential.\\
% </latex>
%% 
% <latex>
% \end{enumerate}
% \section{Integrate and Fire Model (38 pts)}
% You may find it useful to read pg.\ 162-166 of Dayan and Abbott for this section. The general differential equation for the integrate and fire model is
% \[ \tau_m\frac{dV}{dt} = V_m - V(t) + R_m I_e(t) \]
% where $\tau_m = 10\, \rm ms$ is the membrane time constant, describing how fast the current is leaking through the membrane, $V_m$ in this case is constant and represents the resting membrane potential (which you have already calculated in question 1.3.a), and $V(t)$ is the actual membrane potential as a function of time. $R_m = 10^7\, \Omega$ is the constant total membrane resistance, and $I_e(t)$ is the fluctuating incoming current. Here, we do not explicitly model the action potentials (that's Hodgkin-Huxley) but instead model the neuron's behavior leading up and after the action potential.
% </latex>

%%
% <latex>
% Use a $\Delta t = 10\, \rm \mu s$ ($\Delta t$ is the discrete analog of the continuous $dt$). Remember, one strategy for modeling differential equations like this is to start with an initial condition (here, $V(0)=V_m$), then calculate the function change (here, $\Delta V$, the discrete analog to $dV$) and then add it to the function (here, $V(t)$) to get the next value at $t+\Delta t$. Once/if the membrane potential reaches a certain threshold ($V_{th} = -50\, \rm mV$), you will say that an action potential has occurred and reset the potential back to its resting value.
% \begin{enumerate}
%  \item Model the membrane potential with a constant current injection (i.e., $I_e(t) = I_e = 2 {\rm nA}$). Plot your membrane potential as a function of time to show at least a handful of ``firings.'' (8 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% Subsituting the values provided, we can calculate membrane potential for
% given constant current injection.
% </latex>

%%
%Using paramters provided in the question
tau_m_s = 10*1e-3;
V_m_rest_V = -63.2*1e-3; %Value from 1.3a
R_m_ohm = 1e7; 
I_e_A = 2*1e-9;
del_t_s = 10*1e-6 ;
V_m_t_V= [];
V_m_t_curr_V = V_m_rest_V; %Initial V_m_t = V_m
V_m_thresh_V = -50*1e-3;

duration_us = 60000;
spike_curr = 0;

for i=0:10:duration_us
    
    V_m_t_V = [V_m_t_V, V_m_t_curr_V];
    del_V_V = del_t_s/tau_m_s*(V_m_rest_V-V_m_t_curr_V+R_m_ohm*I_e_A);
    V_m_t_curr_V = V_m_t_curr_V+del_V_V;
    %Resetting the membrane potential when it crosses the threshold
    if V_m_t_curr_V > V_m_thresh_V
        V_m_t_curr_V = V_m_rest_V;       
        spike_new = i-spike_curr;
        spike_curr = i;
    end 
    
end

spike_interval_ms = spike_new/1e3;
%%
%Plotting the voltage
figure();
t_us = 0:10:duration_us;
t_ms = t_us/1e3;
V_m_t_mV = V_m_t_V*1e3;
plot(t_ms,V_m_t_mV, 'Linewidth', 1.5)
hold on 
yline(V_m_thresh_V*1e3, ':', 'V_{th}', 'LineWidth', 1)
title(['Membrane potential for I_e = ', num2str(I_e_A),'A'])
xlabel('Time (ms)')
ylabel('Membrane Potential (mV)')
ylim([-65,-48])
hold off


%%
% <latex>
%  \item Produce a plot of firing rate (in Hz) versus injection current, over the range of 1-4 nA. (4 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% To calculate firing rate, we know that, \\
% \[
% $r_{sis}$ = 
% \begin{cases}
%   0 & \text{if $R_mI_e\ \leg \ V_{th} -E_l$} \\
%   \frac{1}{t_isi} = (\tau_m ln(\frac{R_mI_e\ +E_L\ -V_{reset}}{R_mI_e\ +E_L\ - V_{th}}))^{-1} & \text{otherwise}
% \end{cases}
% \]
% </latex>

%%
%Using paramters provided in the question
tau_m_s = 10*1e-3;
V_m_reset_V = V_m_rest;
R_m_ohm = 1e7; 
E_l_V = V_m_reset_V;
V_m_thresh_V = -50*1e-3;

I_e_A = (1:0.1:4)*1e-9;
r_isi_hz = [];

for i=I_e_A
    t_isi_curr_s = tau_m_s*log((R_m_ohm*i + E_l_V - V_m_reset_V)/(R_m_ohm*i + E_l_V - V_m_thresh_V));
    if (R_m_ohm*i <=  (V_m_thresh_V-E_l_V))
        r_isi_curr_hz = 0;
    else
        r_isi_curr_hz = 1/t_isi_curr_s;
    end
    r_isi_hz = [r_isi_hz, r_isi_curr_hz ]; 
end


%%
% figure();
plot(I_e_A,r_isi_hz, 'o-', 'Linewidth', 1.5)
title('Firing rate vs Injection current ')
xlabel('Injection Current (A)')
ylabel('Firing Rate (Hz)')

%%
% <latex>
%  \item \texttt{I521\_A0002\_D001} contains a dynamic current injection in nA. Plot the membrane potential of your neuron in response to this variable injection current. Use Matlab's \texttt{subplot} function to place the plot of the membrane potential above the injection current so that they both have the same time axis. (Hint: the sampling frequency of the current injection data is different from the sampling frequency ($\frac{1}{\Delta t}$) that we used above.) (4 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% </latex>

%%
%Getting I521_A0002_D001 data
%Adding workspace path
addpath(genpath('/Users/jalpanchal/git/be521'));

%create session
% dataset I521 A0002 D001.
session = IEEGSession('I521_A0002_D001', 'jalpanchal', 'jal_ieeglogin.bin');

%fetch detals of data
duration_s = session.data(1).rawChannels(1).get_tsdetails.getDuration/1e6;
sampling_frequency_hz = session.data.sampleRate;
dyn_curr_nA = (session.data.getvalues(0, duration_s * 1e6, 1))';

%%
%Calculating the membrane potential for the provided dynamic injection
%current

tau_m_s = 10*1e-3;
V_m_rest_V = -63.2*1e-3; %Value from 1.3a
R_m_ohm = 1e7; 
del_t_s = 1/sampling_frequency_hz;
V_m_t_V= [];
V_m_t_curr_V = V_m_rest_V; %Initial V_m_t = V_m
V_m_thresh_V = -50*1e-3;


for i=dyn_curr_nA
    i_A = i*1e-9;
    V_m_t_V = [V_m_t_V, V_m_t_curr_V];
    del_V_V = del_t_s/tau_m_s*(V_m_rest_V-V_m_t_curr_V+R_m_ohm*i_A);
    V_m_t_curr_V = V_m_t_curr_V+del_V_V;
    %Resetting the membrane potential when it crosses the threshold
    if V_m_t_curr_V > V_m_thresh_V
        V_m_t_curr_V = V_m_rest_V;
    end 
    
end
%%
%Plotting values
fs = sampling_frequency_hz;
len = duration_s;
t = 0:1/fs:len-1/fs;
x = dyn_curr_nA;

figure();
V_m_t_mV = V_m_t_V*1e3;
ax1 = subplot(2,1,1);
plot(t,V_m_t_mV, 'Linewidth', 1.5)
hold on 
yline(V_m_thresh_V*1e3, ':', 'V_{th}', 'LineWidth', 1)
title(['Membrane potential for dynamic injection current'])
% xlabel('Time (s)')
ylabel('Membrane Potential (mV)')
ylim([-65,-45])
set(ax1,'xticklabel',[]);
ax1.Position = [0.1 0.5 0.8 0.4];
hold off

ax2 = subplot(2,1,2);
plot(t, x, 'Color', 'black', 'Linewidth', 1.5);
ylabel('Injection Current (nA)');
xlabel('Time (sec)');
ax2.Position = [0.1 0.08 0.8 0.4];


%%
% <latex>
%  \item Real neurons have a refractory period after an action potential that prevents them from firing again right away. We can include this behavior in the model by adding a spike-rate adaptation conductance term, $g_{sra}(t)$ (modeled as a potassium conductance), to the model
%  \[ \tau_m\frac{dV}{dt} = V_m - V(t) - r_m g_{sra}(t)(V(t)-V_K)+ R_m I_e(t) \]
%  where \[ \tau_{sra}\frac{dg_{sra}(t)}{dt} = -g_{sra}(t), \]
%  Every time an action potential occurs, we increase $g_{sra}$ by a certain constant amount, $g_{sra} = g_{sra} + \Delta g_{sra}$. Use $r_m \Delta g_{sra} = 0.06$. Use a conductance time constant of $\tau_{sra} = 100\, \rm ms$, a potassium equilibrium potential of $V_K = -70\, \rm mV$, and $g_{sra}(0) = 0$. (Hint: How can you use the $r_m \Delta g_{sra}$ value to update voltage and conductance separately in your simulation?)
% </latex>

%%
% <latex>
%  \begin{enumerate}
%   \item Implement this addition to the model (using the same other parameters as in question 2.1) and plot the membrane potential over 200 ms. (8 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% Here, to account for spike-rate adaption, we will increment the
% $r_m g_{sra}$ term by 0.06 after every action potential and continue
% decaying it at the the rate of \frac{-$r_m g_{sra}$}{\tau_sra} till it
% reaches 0. \\
% </latex>

%%
%Using paramters provided in the question
tau_m_s = 10*1e-3;
V_m_rest_V = -63.2*1e-3; %Value from 1.3a
V_K_V = -70*1e-3; 
R_m_ohm = 1e7; 
I_e_A = 2*1e-9;
del_t_s = 10*1e-6 ;
V_m_t_V= [];
V_m_t_curr_V = V_m_rest_V; %Initial V_m_t = V_m_rest
V_m_thresh_V = -50*1e-3;

tau_sra_s = 100*1e-3;

% As r_m is a constant we will treat r_m * g_sra as a single term
r_m_g_sra = 0;

spike_time_stamp_us = [];

duration_us = 500000;

for i=0:10:duration_us
    
    V_m_t_V = [V_m_t_V, V_m_t_curr_V];
    addi_current = r_m_g_sra*(V_m_t_curr_V-V_K_V);
    del_V_V = del_t_s/tau_m_s*(V_m_rest_V-V_m_t_curr_V+R_m_ohm*I_e_A-addi_current);
    V_m_t_curr_V = V_m_t_curr_V+del_V_V;
    
    %Resetting the membrane potential when it crosses the threshold
    if V_m_t_curr_V > V_m_thresh_V
        V_m_t_curr_V = V_m_rest_V;      
        %After an action potential, r_m*g_sra increases by 0.06
        r_m_g_sra = r_m_g_sra + 0.01;
        
        spike_time_stamp_us = [spike_time_stamp_us, i];
    end 
    
    %While r_m*g_sra is >0 it decays back to zero
    if r_m_g_sra >0        
        del_r_m_g_sra = -del_t_s/tau_sra_s*(r_m_g_sra);
        r_m_g_sra = r_m_g_sra + del_r_m_g_sra;
    end
end

%%
% Plotting the voltage
figure();
t_us = 0:10:duration_us;
t_ms = t_us/1e3;
V_m_t_mV = V_m_t_V*1e3;
plot(t_ms,V_m_t_mV, 'Linewidth', 1.5)
hold on 
yline(V_m_thresh_V*1e3, ':', 'V_{th}', 'LineWidth', 1)
title(('Membrane potential  with spike-rate adaption'))
xlabel('Time (ms)')
ylabel('Membrane Potential (mV)')
ylim([-65,-48])
xlim([0,200])
hold off
%%
% <latex>
%   \item Plot the inter-spike interval (the time between the spikes) of all the spikes that occur in 500 ms. (2 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\ 
% </latex>

%%
%Calculataing interval between spikes
spike_interval_ms = diff(spike_time_stamp_us)/1e3;
plot(spike_interval_ms, 'o-', 'Linewidth', 1.5);
title(('Inter-spike interval from i to i+1^{th} Spike for spike-rate adaption'))
xlabel('Spike Interval index')
ylabel('Interval duration (ms)')
%%
% <latex>
%   \item Explain how the spike-rate adaptation term we introduced above might be contributing to the behavior you observe in 2.4.b. (2 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% Without the spike-rate adaption, we saw that the spike interval was a
% constant value of 10.78 ms for a I_e = 2nA. \\ The spike-rate adaption
% causes the inter-spike distance to lengthen over time and cause it to
% settle at a stady state value eventually. This is achieved by modeling a 
% conductance with fast recovery time and large increment after
% an action potential. The large increment causes the neuron to clamp at
% E_K, thus temporarily preventing further firing and producing an absolute
% refractory period. As this conductance decays to 0, firing will be
% possible but initially less likely, producing a relative refractory
% period. When the recovery is completed, normal firing can resume.\\
% </latex>

%%
% <latex>
%  \end{enumerate}
%  \item Pursue an extension of this basic integrate and fire model. A few ideas are: implement the Integrate-and-Fire-or-Burst Model of Smith et al.\ 2000 (included); implement the Hodgkin-Huxley model (see Dayan and Abbot, pg.\ 173); provide some sort of interesting model of a population of neurons; or perhaps model what an electrode sampling at 200 Hz would record from the signal you produce in question 2.3. Feel free to be creative. 
%  We reserve the right to give extra credit to particularly interesting extensions and will in general be more generous with points for more difficult extensions (like the first two ideas), though it is possible to get full credit for any well-done extension.
%   \begin{enumerate}
% 	\item Briefly describe what your extension is and how you will execute it in code. (6 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% I have attemped to implement the neuron's action potential based on the
% Hodgekin-Huxley model. \\
% The equations used for the implementation are :
% \begin{align*}
% c_m \frac{dV}{dt} = -i_m + \frac{I_e}{A} .\\
% \end{align*}
% Where, i_m is written as a sum of a leakage current, a delayed-rectified
% K^+ current and a transient Na^+ current, as given by:
% \begin{align*}
% i_m = $\bar{g_L}$(V-E_L) + $\bar{g_K}$n^4(V-E_K) +
% $\bar{g_{Na}}$m^3h(V-E_{Na}).\\
% \end{align*}
% The gating the equation for the gating constants used are [1], [2]:
% \begin{align*}
% \frac{dn}{dt}= \alpha_n(V_m)(1-n)-\beta_n(V_m)n \\
% \frac{dm}{dt}= \alpha_m(V_m)(1-m)-\beta_m(V_m)m \\
% \frac{dh}{dt}= \alpha_h(V_m)(1-h)-\beta_h(V_m)h \\ \\
% Where,
% \alpha_n(V_m) = \frac{0.01*(V_m+55)}{1-e^{\frac{-(V_m+55)}{10}}} .\\
% \beta_n(V_m) = 0.125*e^{\frac{-(V_m+65)}{80}} \\ \\
% \alpha_m(V_m) = \frac{0.1(V_m+40)}{1-e^{\frac{-(V+40)}{10}}} \\
% \beta_m(V_m) = 4e^{\frac{-(V_m+65)}{18}} \\ \\
% \alpha_h(V_m) = 0.07e^{\frac{-(V_m+65)}{20}} \\
% \beta_h(V_m) = \frac{1}{1+e^{\frac{-(V_m+35)}{10}}} \\ \\
% \end{align*}
% Using similar method as previous questions, the differential equations
% were implemented using discrete delta change calculation and updating the
% variable. A stepwise increasing i_e was provided to see the change sin
% the gating variables and current. [1][2], \\
% </latex>

%%
% <latex>
% 	\item Provide an interesting figure along with an explanation illustrating the extension. (4 pts)
% </latex>

%%
% <latex>
% \\ Answer: \\
% </latex>

%%
%Using values of constants from DayanAbbot-Chap5_Page 173 and sources [1]
%and [2]
g_L = 0.3; % mS/cm^2
g_K = 36; % mS/cm^2
g_Na = 120; % mS/cm^2

E_L_mV = -54.387;
E_K_mV = -77;
E_Na_mV = 50;

C_m = 1; % uF/cm^2 

t = 0:0.01:450;
del_t_ms = 0.01;

%Injection current
%Steps of 10uA, 20 uA and 40 uA
I_e_uA = 10*(t>50) - 10*(t>150) + 20*(t>200) -20*(t>300) + 40*(t>350) - 40*(t>450);


%Initial values
V_mV = -65;
m = 0.05;
h = 0.6;
n = 0.32;

V_arry_mV = [];
I_L_arry = [];
I_Na_arry = [];
I_K_arry = [];
m_arry = [];
h_arry = [];
n_arry = [];

for i  = 1:45001
    
    V_arry_mV = [V_arry_mV , V_mV];
    
    I_L = g_L*(V_mV-E_L_mV);
    I_L_arry = [I_L_arry, I_L];
    
    I_Na = g_Na*m^3*h*(V_mV-E_Na_mV);
    I_Na_arry = [I_Na_arry, I_Na];
    
    I_K = g_K*n^4*(V_mV-E_K_mV);
    I_K_arry = [I_K_arry, I_K];
    
    
    del_V_mV = del_t_ms/C_m*(I_e_uA(i) -I_L -I_Na -I_K);
    V_mV = V_mV + del_V_mV;
    
    m_arry = [m_arry, m];
    del_m = del_t_ms*((0.1*(V_mV+40)/(1-exp(-(V_mV+40)/10)))*(1-m)-(4*exp(-(V_mV+65)/18))*m);
    m = m + del_m;
    
    h_arry = [h_arry,h];
    del_h = del_t_ms*((0.07*exp(-(V_mV+65)/20))*(1-h)-(1/(1+exp(-(V_mV+35)/10)))*h);
    h = h + del_h;
    
    n_arry = [n_arry, n];
    del_n = del_t_ms*((0.01*((V_mV+55)/(1-exp(-(V_mV+55)/10))))*(1-n)-(0.125*exp(-(V_mV+65)/80))*n);
    n = n + del_n;
    
end

%%
%Plot the graphs
subplot(4,1,1);
plot(t,V_arry_mV, 'LineWidth', 1.1)
title(('Hodgkin-Huxley model for action potential'))
ylabel('Membrane Potential (mV)')

subplot(4,1,2);
plot(t, I_e_uA, 'LineWidth', 1.1)
ylabel('Injection Current (\muA/cm^2)')

subplot(4,1,3);
plot(t, m_arry, 'LineWidth', 1.1)
hold on
plot(t, h_arry, 'LineWidth', 1.1)
plot(t, n_arry, 'LineWidth', 1.1)
hold off
legend('m', 'h', 'n')
ylabel('Gating variables')

subplot(4,1,4);
plot(t, I_L_arry, 'LineWidth', 1.1)
hold on
plot(t, I_Na_arry, 'LineWidth', 1.1)
plot(t, I_K_arry, 'LineWidth', 1.1)
hold off
legend('I_L', 'I_{Na}', 'I_K')
ylabel('Current (\muA)')

xlabel('Time (ms)')





%%
% <latex>
%   \end{enumerate}
% \end{enumerate}
% </latex>


