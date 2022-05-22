# in file “regression.py”
import numpy as np
import scipy.linalg
import scipy

def linear_regression_qr(X, y):
    # add a column [1, ..., 1]^T as 1st column of X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # QR decomposition
    Q, R = scipy.linalg.qr(X)
    col_R = R.shape[1]
    Q = Q[:, :col_R]; R = R[:col_R, :]
    # backward substitution
    b = np.matmul(np.transpose(Q), y)
    N = b.shape[0]
    w = np.empty(shape=b.shape)
    w[N-1] = b[N-1] / R[N-1, N-1]
    for m in range(N-2, -1, -1):
        w[m] = b[m]
    for k in range(m+1, N):
        w[m] -= R[m,k] * w[k]
        w[m] = w[m] / R[m,m]
    return w

def regression_surface(x1, y, x1plot=None, x2plot=None):
    # convert lists to numpy.arrays
    x = np.array(x1); y = np.array(y)
    # convert to column vectors
    M = y.size
    x0 = np.reshape(x[:,0], (M,1))
    x1 = np.reshape(x[:,1], (M,1))
    x2 = np.reshape(x[:,2], (M,1))
    x3 = np.reshape(x[:,3], (M,1))
    x4 = np.reshape(x[:,4], (M,1))
    x5 = np.reshape(x[:,5], (M,1))
    x6 = np.reshape(x[:,6], (M,1))
    x7 = np.reshape(x[:,7], (M,1))
    x8 = np.reshape(x[:,8], (M,1))
    x9 = np.reshape(x[:,9], (M,1))
    x10 = np.reshape(x[:,10], (M,1))
    x11 = np.reshape(x[:,11], (M,1))
    x12 = np.reshape(x[:,12], (M,1))
    x13 = np.reshape(x[:,13], (M,1))#**2
    x14 = np.reshape(x[:,14], (M,1))#**2
    x15 = np.reshape(x[:,15], (M,1))#**2
    x16 = np.reshape(x[:,16], (M,1))#**2
    x17 = np.reshape(x[:,17], (M,1))#**2
    x18 = np.reshape(x[:,18], (M,1))#**2
    x19 = np.reshape(x[:,19], (M,1))#**2
    x20 = np.reshape(x[:,20], (M,1))
    x21 = np.reshape(x[:,21], (M,1))
    x22 = np.reshape(x[:,22], (M,1))
    x23 = np.reshape(x[:,23], (M,1))
    x24 = np.reshape(x[:,24], (M,1))
    x25 = np.reshape(x[:,25], (M,1))
    x26 = np.reshape(x[:,26], (M,1))


    X = np.concatenate((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,
                        x20,x21,x22,x23,x24,x25,x26), axis=1)

    y = np.reshape(y, (M, 1))
    w = linear_regression_qr(X, y)
    if x1plot is None or x2plot is None:
        return w
    else:
        yplot = w[0] + w[1] * x1plot + w[2] * x2plot \
        + w[3] * x1plot**2 + w[4] * x1plot * x2plot \
        + w[5] * x2plot**2
        return yplot


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from regression import regression_surface
    
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
    '''
    w = regression_surface(X, y, x1plot=None, x2plot=None)
    w = np.array(w)
    #print(w)
    new = 0
    '''
    w = scipy.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))
    print(w)
    #print(sol)
    #print(np.dot(X.T,X))
    #w = np.dot(np.dot(X.T,y),np.linalg.inv(np.dot(X.T,X)))
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

    


    


