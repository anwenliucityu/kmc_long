import numpy as np
from line_math import is_parallel,vec_is_parallel,Rx,Ry,Rz

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
    #print(b,R)
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

class W_kink_el:
    def __init__(self,x1,x2,x3,x4,rc,nu,mu,unit_plane,b): # unit_plane from md_input.py
        # Init parameters
        self.nu=nu
        self.mu=mu
        self.rc=rc
        self.b_y=b[0]
        # Determine parallel lines property (dislocation segment or kink), then rotate
        # the coordinate according to book Computer Simulation of Dislocations, Page, 177
        vec=x2-x1 #(or x4-x3, as (x4-x3)||(x2-x1))
        init_array=np.array([x1,x2,x3,x4])
        if vec_is_parallel(vec,unit_plane[0]) == True: # segment line || B plane
            coor=np.inner(init_array,Rz(np.pi/2))
        elif vec_is_parallel(vec,unit_plane[1]) == True: # segment line || Pi1 plane
            rot_angle=np.arctan(unit_plane[1][2]/unit_plane[1][1])
            rot_fun = np.dot(Rz(np.pi/2),Rx(-rot_angle)) # rotate to x-y plane, refer book fig9.4
            coor=np.inner(init_array,rot_fun)
            #print(self.coor)
        elif vec_is_parallel(vec,unit_plane[2]) == True: # segment line || Pi2 plane
            rot_angle=-np.arctan(unit_plane[1][2]/unit_plane[1][1])
            rot_fun = np.dot(Rz(np.pi/2),Rx(-rot_angle)) 
            coor=np.inner(init_array,rot_fun)
            #print(self.coor)
        elif vec_is_parallel(vec,unit_plane[3]) == True: # segment line || P plane
            rot_fun = np.dot(Rz(np.pi/2),Rx(-np.pi/2)) 
            coor=np.inner(init_array,rot_fun)
            #print(self.coor)
        self.coor=coor
    
    def Ra(self,x,y,rc):
        Ra = np.sqrt(x**2+y**2+rc**2)
        return Ra
    
    def f(self,x,y,rc):
        Ra=self.Ra(x,y,rc)
        return x*np.log(Ra+x)-Ra
    
    def g(self,x,y,rc):
        Ra=self.Ra(x,y,rc)
        return (rc**2*Ra)/(2*(y**2+rc**2))
    
    def x(self,i,j):
        return self.coor[j,0]-self.coor[i,0]
    
    def y(self,i,j):
        return self.coor[j,1]-self.coor[i,1]
     
    def calc(self):
        W_hh = self.mu/(np.pi*4)*self.b_y**2/(1-self.nu)*\
            (self.f(self.x(0,3),self.y(0,2),self.rc)-\
             self.f(self.x(1,3),self.y(0,2),self.rc)-\
             self.f(self.x(0,2),self.y(0,2),self.rc)+\
             self.f(self.x(1,2),self.y(0,2),self.rc) )
        return W_hh
        
''' # calculate self-energy, refer to paper non-singular
def W_self(L_vec,rc,mu,nu,bb,bt):
    L = np.linalg.norm(L_vec)
    La = np.sqrt(L**2+rc**2)
    W_self = mu/(np.pi*4*(1-nu))*((bb-nu*bt**2)*L*np.log((La+L)/rc) - \
         (3-nu)/2*bt**2*(La-rc))
    # self-energy
    #if (x4==x2).all() == True and (x3==x1).all()==True:
       # W_self /= 2
    return W_self
'''

class elastic_interaction_energy_ij():
    def __init__(self,x1,x2,x3,x4,b,mu,nu,rc,unit_plane):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.b = b
        self.rc =rc
        self.mu = mu
        self.nu = nu
        self.unit_plane=unit_plane
    
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
        
        if is_parallel(self.x1,self.x2,self.x3,self.x4) == False: # if not parallel
            W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc,bb,bt = self.parameters()
            W1 = W_ns(R1,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W2 = W_ns(R2,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W3 = W_ns(R3,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W4 = W_ns(R4,W0,A1,A2,A2_p,A3,A3_p,A4,A5,v,v_p,t,t_p,tt_p,u,uu,rc)
            W_total = (W1+W2-W3-W4)/160.21766208
            #print(W_total)
        elif (self.x4==self.x2).all() == False and (self.x3==self.x1).all()== False\
            and vec_is_parallel(self.x2-self.x1, np.array([1,0,0])) == True: 
            # calculate elastic interaction only for segments parallel to x direction.
            t,t_p,tt_p,rc,bb,bt = self.parameters()
            W1 = W_ns_para(bt,self.b,R1,self.nu,self.mu,bb,rc,t)
            W2 = W_ns_para(bt,self.b,R2,self.nu,self.mu,bb,rc,t)
            W3 = W_ns_para(bt,self.b,R3,self.nu,self.mu,bb,rc,t)
            W4 = W_ns_para(bt,self.b,R4,self.nu,self.mu,bb,rc,t)
            W_total = (W1+W2-W3-W4)/160.21766208
            #print(W_total)
        elif vec_is_parallel(self.x2-self.x1, np.array([1,0,0])) == False:
            W_total=W_kink_el(self.x1,self.x2,self.x3,self.x4,self.rc,self.nu,self.mu,self.unit_plane,self.b).calc()/160.21766208
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
    #W5 = elastic_interaction_energy_ij(x1,x2,x3,x4,b,mu,nu,rc).calc_elastic_interaction()
    #print(W5)
    unit_plane=[[0,  np.sqrt(3)*a/2, 0],   #B
                [0,  np.sqrt(3)*a/2, c],   #Pi1
                [0, -np.sqrt(3)*a/2, c],   #Pi2
                [0,               0, c]]   #P
    W1 = []
    for i in range(1,100):
        x1 = np.array([0,0,0])
        x2 = np.array([0, 0,1])
        x3 = np.array([i,0,1])
        x4 = np.array([i,0,0])
        #w1=W_kink_el(x1,x2,x3,x4,rc,nu,mu,unit_plane,b).calc()/160.21766208
        w = elastic_interaction_energy_ij(x1,x2,x3,x4,b,mu,nu,rc,unit_plane).calc_elastic_interaction()
        W1.append(w)

    import matplotlib.pyplot as plt
    #plt.plot(range(1,100),W1,label='W1')
    plt.legend()
    plt.show()



