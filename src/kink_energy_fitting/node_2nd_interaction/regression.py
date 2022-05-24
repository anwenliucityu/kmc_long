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
    PP_B =  [ 5.62597246, 18.30996842, 35.40895892]
    PP_P =  [ 1.31253927,  8.66616867, 19.35794728]
    PP_Pi =  [ 8.16444238, 26.86075748, 52.09382775]

    PiPi_B =  [ 3.34258521,  8.06222404, 14.69617094]
    PiPi_P =  [ 3.89700461, 23.56899918, 39.27173704]
    PiPi_Pi1 =  [ 3.62039451, 14.53357431, 30.04730888]
    PiPi_Pi2 =  [ 6.36536716, 20.15393153, 38.52192858]

    BB_B =  [ 6.99898778, 13.33354911, 21.04261698]
    BB_P =  [14.62845244, 28.0312919,  41.16448485]
    BB_Pi =  [15.73117502, 27.72134461, 43.75705646]

    BP_B =  [ 2.82899199,  8.26360242, 15.97674954]
    BP_P =  [ 6.27719034, 19.18752874, 31.32345462]
    BP_Pi =  [15.73117502, 27.72134461, 43.75705646]

    BPi_B  =  [ 6.40928929, 11.34625629, 19.61174584]
    BPi_P =  [ 8.29777051, 19.70833336, 34.81447626]
    BPi_Pi =  [15.73117502, 27.72134461, 43.75705646]
    BPi_Pi2 =  [10.20507735, 25.94580494, 43.12995868]

    diff_PiPi_B  =  [ 5.35902947, 10.27011978, 17.76012722]
    diff_PiPi_P =  [ 5.03696318, 12.72572458, 24.13051448]
    diff_PiPi_Pi =  [ 7.96498366, 21.45418062, 38.84253908]

    PPi_B =  [ 5.67371426,  9.90976838, 17.6461792 ]
    PPi_P =  [ 4.77294985, 14.0337006,  27.28679131]
    PPi_Pi =  [ 4.30691071, 19.11241683, 36.42770598]
    PPi_Pi2 =  [ 8.5504261,  22.89966278, 40.0068594 ]

    y = [PP_B,PP_P,PP_Pi,PiPi_B,PiPi_P,PiPi_Pi1,PiPi_Pi2,BB_B,BB_P,BB_Pi,BP_B,BP_P,BP_Pi,BPi_B,BPi_P,BPi_Pi,BPi_Pi2,diff_PiPi_B,diff_PiPi_P,diff_PiPi_Pi,PPi_B,PPi_P,PPi_Pi,PPi_Pi2]
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
    if os.path.isfile('w1.dat')==False:
        np.savetxt('w1.dat',w)

    new=0
    for i in range(w.shape[0]):
        #if i==0:
        #    new==w[0]
        #else:
        new += w[i]*X[:,i]
    new=new.reshape((24,-1))

    # plot
    y = [PP_B,PP_P,PP_Pi,PiPi_B,PiPi_P,PiPi_Pi1,PiPi_Pi2,BB_B,BB_P,BB_Pi,BP_B,BP_P,BP_Pi,BPi_B,BPi_P,BPi_Pi,BPi_Pi2,diff_PiPi_B,diff_PiPi_P,diff_PiPi_Pi,PPi_B,PPi_P,PPi_Pi,PPi_Pi2]
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

    


    


