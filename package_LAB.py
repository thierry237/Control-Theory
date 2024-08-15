import numpy as np
from package_DBR import *
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.interpolate import interp1d


def LeadLag_RT(MV,Kp,Tlead,Tlag,Ts,PV,PVInit=0,method='EBD'):
    
    """
    LeadLag_RT(MV,Kp,Tlead,Tlag,Ts,PV,PVInit=0,method='EBD')
        The function "LeadLag_RT" needs to be included in a "for or while loop".
        
        :MV: input vector
        :Kp: process gain
        :Tlead: lead time constant [s]
        :Tlag: lag time constant [s]
        :Ts: sampling period [s]
        :PV: output vector
        :PVInit: (optional: default value is 0)
        :method: discretisation method (optional: default value is 'EBD')
            EBD: Euler Backward difference
            EFD: Euler Forward difference
            TRAP: Trapezoïdal method
        
        The function appends a value to the output vector "PV".
        The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
               PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1 + Tlead/Ts)*MV[-1] - (Tlead/Ts)*MV[-2]))
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp*((Tlead/Ts)*MV[-1] + (1 - (Tlead/Ts))*MV[-2]))
            elif method == 'TRAP':
                PV.append((((2-K)/(2+K))*PV[-1]) + ((Kp*K/(2+K))*((2*Tlead/Ts)+ 1))*MV[-1] + ((Kp*K/(2+K))*(1 - (2*Tlead/Ts)))*MV[-2] )            
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1 + Tlead/Ts)*MV[-1] - (Tlead/Ts)*MV[-2]))
    else:
        PV.append(Kp*MV[-1])



def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    '''
    The function "PID_RT" needs to be included in a "for or while loop". 

    :SP: SP (or SetPoint) vector 
    :PV: PV (or Process Value) vector 
    :Man: Man (or Manual controller mode) vector (True or False) 
    :MVMan: MVMan (or Manual value for MV) vector 
    :MVFF: MVFF (or Feedforward) vector 

    :Kc: controller gain 
    :Ti: integral time constant [s] 
    :Td: derivative time constant [s] 
    :alpha: Tfd = alpha*Td where Tfd is the derivative filter time constant [s] 
    :Ts: sampling period [s] 

    :MVMin: minimum value for MV (used for saturation and anti wind-up) 
    :MVMax: maximum value for MV (used for saturation and anti wind-up) 

    :MV: MV (or Manipulated Value) vector 
    :MVP: MVP (or Propotional part of MV) vector 
    :MVI: MVI (or Integral part of MV) vector 
    :MVD: MVD (or Derivative part of MV) vector 
    :E: E (or control Error) vector 

    :ManFF: Activated FF in manual mode (optional: default boolean value is False) 
    :PVInit: Initial value for PV (optional: default value is 0): used if PID_RT is ran first in the squence and no value of PV is available yet. 

    :method: discretisation method (optional: default value is 'EBD') 
        EBD-EBD: EBD for integral action and EBD for derivative action 
        EBD-TRAP: EBD for integral action and TRAP for derivative action 
        TRAP-EBD: TRAP for integral action and EBD for derivative action 
        TRAP-TRAP: TRAP for integral action and TRAP for derivative action 

    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD". The appended values are based on the PID algorithm, the controller mode, and feedforward. Note that saturation of "MV" within the limits [MVMin MVMax] is implemented with anti wind-up. 
        '''

    ''' Initialisation de E'''
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else:   
        E.append(SP[-1] - PV[-1])

    methodI, methodD = method.split('-')
    
    ''' Initialisation de MVI'''
    if(Ti>0):
        if len(MVI) == 0:
            if methodI == 'TRAP':
                MVI.append(0.5*(Kc*Ts/Ti)*E[-1])
            else :
                MVI.append((Kc*Ts/Ti)*E[-1])
        else:
            if methodI == 'TRAP':
                MVI.append(MVI[-1] + (0.5*Kc*Ts/Ti)*(E[-1]+E[-2]))
            else : #EBD
                MVI.append(MVI[-1] + (Kc*Ts/Ti)*E[-1])


    '''Initialisation de MVD : '''
    Tfd = alpha * Td
    if Td > 0:
        if len(MVD) != 0:
            if len(E) == 1:
                if methodD =='TRAP':
                    MVD.append((Tfd-(Ts/2))/(Tfd/(Ts/2))*MVD[-1] + (Kc*Td/(Tfd + (Ts/2)))*(E[-1]))
                else :
                    MVD.append((Tfd / (Tfd + Ts)) *
                            MVD[-1] + ((Kc * Td) / (Tfd + Ts)) * (E[-1]))
            else:
                if methodD =='EBD':
                    MVD.append((Tfd / (Tfd + Ts)) *
                            MVD[-1] + ((Kc * Td) / (Tfd + Ts)) * (E[-1] - E[-2]))
                else : #Trap
                    MVD.append((Tfd-(Ts/2))/(Tfd/(Ts/2))*MVD[-1] + (Kc*Td/(Tfd + (Ts/2)))*(E[-1]-E[-2]))
        else:
            if len(E) == 1:
                if methodD == 'TRAP':
                    MVD.append(+ (Kc*Td/(Tfd + (Ts/2)))*(E[-1]))    
                else : 
                    MVD.append((Kc * Td) / (Tfd + Ts) * (E[-1]))
            else:
                MVD.append((Kc * Td) / (Tfd + Ts) * (E[-1] - E[-2]))

    '''Actualisation de MVP'''
    MVP.append(E[-1] * Kc)
    
    '''Mode manuel et anti-wind-up'''

    if Man[-1]:
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]

    '''Limitation de MV'''
    
    if MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] >= MVMax:
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFF[-1]

    if MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1] <= MVMin:
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFF[-1]

    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])



class Controller:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Kc'] = parameters['Kc'] if 'Kc' in parameters else 0.0
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 0.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['alpha'] = parameters['alpha'] if 'alpha' in parameters else 0.0




def IMC_Tuning(P, C, gamma):
    """
    The function "IMC_Tuning_Process" is only for first and second order systems.
    :P: process object
        :Kp: process gain
        :Tlag1: first (or main) lag time constant [s] used in your process
        :Tlag2: second lag time constant [s] used in your process
        :theta: delay [s] used in your process
    :C: controller object
        :Kc: controller gain
        :Ti: reset time
        :Td: derivative time
    :gamma: constant used to get the closed loop time constant
    
    """
    Tc = gamma *  P.parameters['Tlag1']
    C.parameters['Kc'] = ((P.parameters['Tlag1'] + P.parameters['Tlag2']) / (Tc + P.parameters['theta'])) / P.parameters['Kp']
    C.parameters['Ti'] = P.parameters['Tlag1'] + P.parameters['Tlag2']
    C.parameters['Td'] = (P.parameters['Tlag1']*P.parameters['Tlag2']) / (P.parameters['Tlag1'] + P.parameters['Tlag2'])



def Stability_Margins(P,C,omega, show=True):   

    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})
        
        A delay, two lead time constants and 2 lag constants can be added.
        
        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0  
        
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-2, 2, 10000)".
    :Show: boolean value (optional: default value = True).
        If Show == True, the Bode diagram is show.
        If Show == False, the Bode diagram is NOT show and the complex vector Ps is returned.
    
    The function "Bode" generates the Bode diagram of the process L(s) = P(s)C(s) and returns the complex vector Ls.
    """  
    s = 1j * omega

    # Create params for P
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1
    
    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    Ps = np.multiply(Ps,PLead1)
    Ps = np.multiply(Ps,PLead2)

    # Create params for C
    #Direct method Cs = C.parameters['Kc']*( 1 + (1/(C.parameters['Ti']*s)) + (C.parameters['Td']*s)/(C.parameters['alpha'] * C.parameters['Td'] * s + 1))

    #proportional action P
    P = np.multiply(C.parameters['Kc'],1)
    Cs = P * np.ones_like(s)
    
    #Integrator action I
    I = np.multiply(C.parameters['Kc']*np.ones_like(s), 1/(C.parameters['Ti']*s))
    Cs = np.add(Cs, I)
   
    #Derivative action D
    DGain = C.parameters['Kc'] * np.ones_like(s)
    Td = C.parameters['Td']* s
    alpha = (1 / (C.parameters['alpha'])) * np.ones_like(s)
    Tfd =  ((C.parameters['Td']) * alpha *s +1)

    D = np.multiply(DGain, Td) 
    D = np.divide(D, Tfd)
    Cs = np.add(Cs, D)

    # without numpy Cs = P + I + D

    # Calculate values for Bode diagram of L=P*C
    Ls = np.multiply(Ps, Cs)

    if show == True:
        
        fig, (ax_gain, ax_phase) = plt.subplots(2,1)
        fig.set_figheight(12)
        fig.set_figwidth(22)

        # Gain crossover frequency (wc) and Phase crossover frequency (wu)
        wc_idx = np.argmin(np.abs(np.abs(Ls) - 1))
        wc = omega[wc_idx]
        wu_idx = np.argmin(np.abs((180 / np.pi) * np.angle(Ls) + 180 + 180))
        wu = omega[wu_idx]

        # Phase margin  and Gain margin 
        phase_margin = 180 + (180 / np.pi) * np.angle(Ls[wc_idx])
        gain_margin = 1 / np.abs(Ls[wu_idx])

        print("Gain crossover frequency (wc) :", wc)
        print("Phase crossover frequency (wu) :", wu)
        print("Phase margin:", phase_margin)
        print("Gain margin :", gain_margin)
        #-------------------------
        #Compute the gain and phase of Ls
         # Gain part
        # Find the indices where the gain curve crosses zero
        zero_crossings = np.where(np.diff(np.sign(20*np.log10(np.abs(Ls)))))[0]
        # Interpolate the corresponding omega values
        omega_intersect = np.interp(0, 20*np.log10(np.abs(Ls[wc])), omega[zero_crossings])
        print("Gain crossover frequency :", omega_intersect)
        #-------------------------
        # Phase part
        # Find the indices where the phase curve is closest to -180 degrees
        phase_diff = np.abs((180/np.pi)*np.angle(Ls) + 180 + 180)
        min_phase_diff_idx = np.argmin(phase_diff)
    
        # Get the value of omega corresponding to this index
        omega_intersect_phase = omega[min_phase_diff_idx]

        # Find the phase at the index where the difference is minimal
        phase_intersect = (180/np.pi)*np.angle(Ls[min_phase_diff_idx])
        
        # Gain part
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ls)),label=r'$L(s)$')   
        gain_min = np.min(20*np.log10(np.abs(Ls)/5))
        gain_max = np.max(20*np.log10(np.abs(Ls)*5))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.axhline(0, color='r', linestyle='--', linewidth=1)
        ax_gain.vlines(omega_intersect,gain_min,0, color = 'green', linestyle = '--')
         # Calculate |L(jw)| for all frequencies omega
        Ls_abs = np.abs(Ls)
        # Find the index where |L(jw)| is closest to 1
        idx_closest_to_1 = np.argmin(np.abs(Ls_abs - 1))

        # Get the corresponding frequency value
        omega_u = omega[idx_closest_to_1]

       

        # Find the index corresponding to omega_u in the omega vector
        index_omega_u = np.where(omega == omega_intersect_phase)[0][0]

        # Get the gain amplitude at omega_u
        gain_at_omega_u = 20 * np.log10(np.abs(Ls[index_omega_u]))

        # Display the gain at omega_u
        print("Gain d'amplitude à arg[L(jw)] = -180° :", gain_at_omega_u)

        # Plot the corresponding point on the amplitude curve
        ax_gain.plot(omega_intersect_phase, gain_at_omega_u, 'go', markersize=8, label='wu')
        ax_gain.vlines(omega_intersect_phase, gain_at_omega_u, 0, colors='black', linestyles='-', linewidth=3)
         # value of the phase margin on the graph
        ax_gain.annotate(f'Margin Gain : {gain_margin:.2f} dB',
                        xy=(omega_intersect_phase, gain_at_omega_u), 
                        xytext=(10, -60),  
                        textcoords='offset points', 
                        color='red', fontsize=14,
                        arrowprops=dict(facecolor='red', arrowstyle='->'))  

        # Plot the point (omega_u, |L(jw)|) on the Bode plot
        ax_gain.plot(omega_u, 20*np.log10(np.abs(Ls[idx_closest_to_1])), 'go')
        ax_gain.set_ylabel('Amplitude' + '\n $|L(j\omega)|$ [dB]')
        ax_gain.set_title('Bode plot of L(s) = P(s)C(s)')
        ax_gain.legend(loc='best')
    
        # Phase part
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ls)),label=r'$L(s)$')   
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ls))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ls))) + 10
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')   
        ax_phase.axhline(-180, color='r', linestyle='--', linewidth=1)
       

        # Calculate the phase of L(jw_u) at the crossover frequency omega_u
        phase_at_omega_u = (180/np.pi) * np.angle(Ls[idx_closest_to_1])

        # Print the value of phase_at_omega_u
        print("Phase at gain crossover frequency:", phase_at_omega_u)
        ax_phase.plot(omega_u, phase_at_omega_u, 'go')
        phase_line = phase_at_omega_u + 180
        ax_phase.vlines(omega_u, phase_at_omega_u, -180, color='green', linestyle='-', linewidth=3)

        # value of the phase margin on the graph
        ax_phase.annotate(f'Margin Phase : {phase_line:.2f}°',
                        xy=(omega_u, -180), 
                        xytext=(15, -30),  
                        textcoords='offset points', 
                        color='red', fontsize=14,
                        arrowprops=dict(facecolor='red', arrowstyle='->'))  
        
        ax_phase.plot(omega_intersect_phase, phase_intersect, 'ro')
        ax_phase.vlines(omega_intersect_phase, ph_max, -180, color='red', linestyle='--')
        ax_phase.set_ylabel('Phase' + '\n $\,$'  + r'$\angle L(j\omega)$ [°]')
        ax_phase.legend(loc='best')
        ax_gain.grid(True)
        ax_phase.grid(True)       
            
    else:
         return Ls


