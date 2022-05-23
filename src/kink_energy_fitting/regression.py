# in file “regression.py”
import numpy as np
import scipy.linalg
import scipy
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    ##############input    
    x1 = [1,2,3]
    # fit BB
    PP_B =  [ 4.81413361, 15.24821279, 28.98135829]
    PP_P =  [-0.6694105,   0.86405405,  2.13387825]
    PP_Pi =  [ 5.39953219, 16.29121178, 29.28711287]

    PiPi_B =  [2.53074636, 5.00046842, 8.26857031]
    PiPi_P =  [ 1.91505484, 15.76688456, 22.04766801]
    PiPi_Pi1 =  [0.85548432, 3.96402861, 7.240594  ]
    PiPi_Pi2 =  [ 3.60045697,  9.58438583, 15.7152137 ]

    BB_B  =    [ 9.28572729, 16.583122  , 19.87285538]
    BB_P =  [12.64650267, 20.22917728, 23.94041582]
    BB_Pi =  [12.96626483, 17.15179891, 20.95034158]

    BP_B  =    [2.01715314, 5.20184679, 9.54914891]
    BP_P =  [ 4.29524057, 11.38541412, 14.09938559]
    BP_Pi =  [12.96626483, 17.15179891, 20.95034158]

    BPi_B  =   [ 5.59745044,  8.28450067, 13.18414521]
    BPi_P =  [ 6.31582074, 11.90621874, 17.59040723]
    BPi_Pi =  [ 7.44016716, 15.37625924, 20.3232438 ]
    BPi_Pi2 = [12.96626483, 17.15179891, 20.95034158]

    dif_PiPi_B  =  [ 4.54719062,  7.20836416, 11.33252659]
    dif_PiPi_P =  [3.05501341, 4.92360995, 6.90644545]
    dif_PiPi_Pi =  [ 5.20007347, 10.88463492, 16.0358242 ]

    PPi_B  =   [ 4.86187541,  6.84801276, 11.21857857]
    PPi_P =  [ 2.79100008,  6.23158598, 10.06272228]
    PPi_Pi =  [ 1.54200052,  8.54287113, 13.6209911 ]
    PPi_Pi2 =  [ 5.78551591, 12.33011708, 17.20014452]

    y = [PP_B,PP_P,PP_Pi,PiPi_B,PiPi_P,PiPi_Pi1,PiPi_Pi2,BB_B,BB_P,BB_Pi,BP_B,BP_P,BP_Pi,BPi_B,BPi_P,BPi_Pi,BPi_Pi2,dif_PiPi_B,dif_PiPi_P,dif_PiPi_Pi,PPi_B,PPi_P,PPi_Pi,PPi_Pi2]
    y = np.array(y).reshape((-1,))
    #print(y)
    X = np.loadtxt('input.txt')
    X = np.delete(X,[13,14,15,16,17,18,19],axis=1)
    print(X)

    #X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X[:,20:] = (2*X[:,20:])**2

    ###############################

    w  = scipy.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))
    #print(w)
    # write w
    if os.path.isfile('w.dat')==False:
        np.savetxt('w.dat',w)

    new=0
    for i in range(w.shape[0]):
        #if i==0:
        #    new==w[0]
        #else:
        new += w[i]*X[:,i]
    new=new.reshape((24,-1))

    # plot
    y = [PP_B,PP_P,PP_Pi,PiPi_B,PiPi_P,PiPi_Pi1,PiPi_Pi2,BB_B,BB_P,BB_Pi,BP_B,BP_P,BP_Pi,BPi_B,BPi_P,BPi_Pi,BPi_Pi2,dif_PiPi_B,dif_PiPi_P,dif_PiPi_Pi,PPi_B,PPi_P,PPi_Pi,PPi_Pi2]
    y = np.array(y)
    name=['PP(B kink)','PP(P kink)',r'PP($\pi$ kink)',r'$\pi \pi$(B kink)',r'$\pi \pi$(P kink)',
            r'$\pi \pi$($\pi$ kink)', r'$\pi \pi$($\pi^\prime$ kink)','BB(B kink)','BB(P kink)',r'BB($\pi$ kink)','BP(B kink)',
            'BP(P kink)',r'BP($\pi$ kink)$',r'B$\pi$(B kink)',r'B$\pi$(P kink)',
            r'B$\pi$($\pi$ kink)',r'B$\pi$($\pi^\prime$ kink)',r'$\pi \pi^\prime$(B kink)',
            r'$\pi \pi^\prime$(P kink)',r'$\pi \pi^\prime$($\pi$ kink)',
            r'P$\pi$(B kink)',r'P$\pi$(P kink)',r'P$\pi$($\pi$ kink)',r'P$\pi$($\pi^\prime$ kink)']
    for i in range(y.shape[0]):
        fig,ax = plt.subplots(figsize=(6,5))
        plt.title(f'{name[i]}', fontsize=18)
        plt.plot([1,2,3],y[i], 'o-', linewidth=4,markersize=15)
        plt.plot([1,2,3],new[i], '--',linewidth=4)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.xaxis.get_offset_text().set_fontsize(16)
        #plt.savefig(f'{i}.png')
        plt.close()
        #plt.show()
    #print(new)

    


    


