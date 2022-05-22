from os import wait4
import numpy as np

def W_ns(R,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc):
    Ra = np.sqrt(np.dot(R,R)+rc**2)
    Rv_p = np.dot(R,v_p)
    Rt = np.dot(R,t)
    Rt_p = np.dot(R,t_p)
    Ru = np.dot(R,u)
    Rv = np.dot(R,v)
    v_sqrt = np.sqrt(uu*rc**2+Ru**2)
    W_ns_W0 = (A1-A2_p)*Rv_p*np.log(Ra+Rt_p)+\
              A3_p*Ru*np.log(Ra+Rt_p)+\
              (A1-A2)*Rv*np.log(Ra+Rt)+\
              A3*Ru*np.log(Ra+Rt) + A4*Ra +\
              (A1-A5)*(2*Ru**2+uu*rc**2)/v_sqrt*\
               np.arctan(((1+tt_p)*Ra+np.dot(R,t+t_p))/v_sqrt)
    W_ns = W_ns_W0*W0
    return W_ns

def W_ns_para(bt,b,R,nu,mu,bb,rc,t):
    bR = np.dot(b,R)
    Ra = np.sqrt(np.dot(R,R)+rc**2)
    Rt = np.dot(R,t)
    #print((bR-Rt*bt)*(bR-Rt*bt)*Ra/(Ra**2-Rt**2))
    W_ns_W0 = (2*bt*bR - ((2-nu)*bt**2+bb)*Rt)*np.log(Ra+Rt) + \
              ((1-nu)*bt**2)*Ra - \
              (bR-Rt*bt)*(bR-Rt*bt)*Ra/(Ra**2-Rt**2)+ \
              rc**2*((1+nu)*bt**2-2*bb)*Ra/(2*(Ra**2-Rt**2))
    W0 = mu/(np.pi*4*(1-nu))
    W_ns = W_ns_W0*W0

    return W_ns

def is_parallel(x1,x2,x3,x4):
    vec1 = x2-x1
    vec2 = x4-x3
    if abs(vec1[0]*vec2[1]-vec1[1]*vec2[0] + vec1[0]*vec2[2]-vec1[2]*vec2[0] + vec1[1]*vec2[2]-vec1[2]*vec2[1]) <1e-4:
        return True                                                                                                     
    else:
        return False

def vec_is_parallel(vec1, vec2):
    if abs(vec1[0]*vec2[1]-vec1[1]*vec2[0] + vec1[0]*vec2[2]-vec1[2]*vec2[0] + vec1[1]*vec2[2]-vec1[2]*vec2[1]) <1e-4:
        return True
    else:
        return False

def W_self(L_vec,rc,mu,nu,bb,bt):
    L = np.linalg.norm(L_vec)
    La = np.sqrt(L**2+rc**2)
    W_self = mu/(np.pi*4*(1-nu))*((bb-nu*bt**2)*L*np.log((La+L)/rc) - \
         (3-nu)/2*bt**2*(La-rc))
    # self-energy
    #if (x4==x2).all() == True and (x3==x1).all()==True:
       # W_self /= 2
    return W_self

class elastic_interaction_energy_ij():
    def __init__(self,x1,x2,x3,x4,b,mu,nu,rc):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.b = b
        self.rc =rc
        self.mu = mu
        self.nu = nu
    
    def parameters(self):
        t = (self.x2-self.x1)/np.linalg.norm(self.x2-self.x1)
        t_p = (self.x4-self.x3)/np.linalg.norm(self.x4-self.x3)
        u = np.cross(t,t_p)
        v = np.cross(u,t)
        v_p = np.cross(t_p,u)
        #Ra = np.sqrt(np.dot(R,R)+self.rc**2)

        bb = np.dot(self.b, self.b)
        bt = np.dot(self.b, t)
        bt_p = np.dot(self.b, t_p)
        tt_p = np.dot(t,t_p)
        bu = np.dot(self.b,u)
        bv = np.dot(self.b,v)
        bv_p = np.dot(self.b,v_p)
        uu = np.dot(u,u)
        
        if is_parallel(self.x1,self.x2,self.x3,self.x4) == False:
            W0 = self.mu/(np.pi*4*(1-self.nu)*uu)
            A1 = (1+self.nu)*bt*bt_p
            A2 = (bb+bt**2)*tt_p
            A2_p = (bb+bt_p**2)*tt_p
            A3 = 2*bu*bv*tt_p/uu
            A3_p = 2*bu*bv_p*tt_p/uu
            A4 = (bt*bv+bt_p*bv_p)*tt_p
            A5 = 2*bu**2*tt_p/uu
            return W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,self.rc,bb,bt
        
        else:
            return t,t_p,tt_p,self.rc,bb,bt

    def calc_elastic_interaction(self):
        R1 = self.x4-self.x2
        R2 = self.x3-self.x1
        R3 = self.x4-self.x1
        R4 = self.x3-self.x2
        
        if is_parallel(self.x1,self.x2,self.x3,self.x4) == False:
            W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc,bb,bt = self.parameters()
            W1 = W_ns(R1,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W2 = W_ns(R2,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W3 = W_ns(R3,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W4 = W_ns(R4,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W_total = (W1+W2-W3-W4)/160.21766208
            #print(W_total)
        elif (self.x4==self.x2).all() == False or (self.x3==self.x1).all()== False: # calculate elastic interaction only.
        #else:
            #print(self.x1,self.x2,self.x3,self.x4)
            t,t_p,tt_p,rc,bb,bt = self.parameters()
            '''
            if (self.x4==self.x2).all() == True and (self.x3==self.x1).all()==True:
                print(2)
                #print([self.x4, self.x2])
                L1 = self.x2-self.x1
                L2 = self.x4-self.x3
                W5 = W_self(L1,rc,self.mu,self.nu,bb,bt)
                W6 = W_self(L2,rc,self.mu,self.nu,bb,bt)
                W_total = (W5+W6)/160.21766208
                print(W_total, self.x1, self.x2, self.x3, self.x4,)
            else:
                print(3)
            '''
            W1 = W_ns_para(bt,self.b,R1,self.nu,self.mu,bb,rc,t)
            W2 = W_ns_para(bt,self.b,R2,self.nu,self.mu,bb,rc,t)
            W3 = W_ns_para(bt,self.b,R3,self.nu,self.mu,bb,rc,t)
            W4 = W_ns_para(bt,self.b,R4,self.nu,self.mu,bb,rc,t)
            W_total = (W1+W2-W3-W4)/160.21766208
            #print(W_total)
        else:
            W_total = 0

        return W_total
        

if __name__ =='__main__':
    a = 2.9365976437421808
    c = 4.6410202908664067
    C13 = 83.235
    C44 = 54.91 #GPa
    b = np.array([a,0,0])
    rc = 10*np.linalg.norm(b)
    mu = C44
    nu = C13/(2*(C13+C44))

    x1 = np.array([0,0,0])
    x2 = np.array([0, 5.08633632, 4.64102029])
    x3 = np.array([5.87319529*5*3,0,0])
    x4 = np.array([5.87319529*5*3,5.08633632, 4.64102029])

    x1 = np.array([0,0,0])
    x2 = np.array([5,0,0])
    x3 = np.array([5,0,0])
    x4 = np.array([10,0,0])
    W1  = elastic_interaction_energy_ij(x1,x2,x3,x4,b,mu,nu,rc).calc_elastic_interaction()

    x1 = np.array([0,0,0])
    x2 = np.array([5,0,0])
    x3 = np.array([0,0,0])
    x4 = np.array([5,0,0])
    W2 = elastic_interaction_energy_ij(x1,x2,x3,x4,b,mu,nu,rc).calc_elastic_interaction()
    
    x1 = np.array([0,0,0])
    x2 = np.array([0,10,0])
    x3 = np.array([6,0,0])
    x4 = np.array([6,10,0])
    W3 = elastic_interaction_energy_ij(x1,x2,x3,x4,b,mu,nu,rc).calc_elastic_interaction()

    x1 = np.array([52.85875759 , 0.         , 0.        ])
    x2 = np.array([52.85875759 ,15.25900896 ,13.92306087])
    x3 = np.array([23.49278115  ,0.      ,    0.        ])
    x4 = np.array([23.49278115, 15.25900896 ,13.92306087])
    W4 = elastic_interaction_energy_ij(x1,x2,x3,x4,b,mu,nu,rc).calc_elastic_interaction()
    print(W4)

    #print(W1+W2,W3/2)
    print(W3)



