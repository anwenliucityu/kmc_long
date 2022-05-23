import numpy as np
import multiprocessing as ml
from elastic_energy import elastic_interaction_energy_ij

def generate_dislocation_line(num_hon_seg, init_core=3, B=0,Pi=0,P=0,a=2.9365976437421808,c=4.6410202908664067):
    # Creat an empty array.
    disl_hon_seg = np.zeros(shape=(3,num_hon_seg))
    # Initiate core structure to be P core. 0=B, 1=Pi1, 2=Pi2, 3=P
    disl_hon_seg[0,:] = init_core
    disl_ver_seg = np.zeros(shape=(1,num_hon_seg,2,2))
    if B != 0:
        disl_hon_seg[1,int(num_hon_seg/2):] += B*np.sqrt(3)*a
        disl_ver_seg[0,int(num_hon_seg/2)-1:num_hon_seg-1,1,0] += B*np.sqrt(3)*a/2
        disl_ver_seg[0,int(num_hon_seg/2):int(num_hon_seg),0,0] += B*np.sqrt(3)*a/2
    if P != 0:
        disl_hon_seg[2,int(num_hon_seg/2):] += P*c
        disl_ver_seg[0,int(num_hon_seg/2)-1:num_hon_seg-1,1,1] += P*c
        disl_ver_seg[0,int(num_hon_seg/2):int(num_hon_seg),0,1] += P*c
    if Pi != 0:
        disl_hon_seg[1,int(num_hon_seg/2):] += Pi*np.sqrt(3)*a
        disl_hon_seg[2,int(num_hon_seg/2):] += Pi*c
        disl_ver_seg[0,int(num_hon_seg/2)-1:num_hon_seg-1,1,0] += Pi*np.sqrt(3)*a/2
        disl_ver_seg[0,int(num_hon_seg/2):int(num_hon_seg),0,0] += Pi*np.sqrt(3)*a/2
        disl_ver_seg[0,int(num_hon_seg/2)-1:num_hon_seg-1,1,1] += Pi*c
        disl_ver_seg[0,int(num_hon_seg/2):int(num_hon_seg),0,1] += Pi*c

    return disl_hon_seg, disl_ver_seg

def switch_hon_seg_coor(disl_hon_seg, seg_len): # switch state to coor
    hon_x_coor = np.array(range(1,disl_hon_seg.shape[1]+1)) * seg_len
    hon_seg_coor = np.empty(shape=(disl_hon_seg.shape[1],3))
    hon_seg_coor[:,0] = hon_x_coor
    hon_seg_coor[:,1:3] = disl_hon_seg[1:3,:].T
    return hon_seg_coor

def switch_init_ver_seg_coor(disl_ver_seg, seg_len): # switch state to coor
    hon_x_coor = np.array(range(0,disl_ver_seg.shape[1])).reshape(-1,1) * seg_len
    hon_x_coor = np.append(hon_x_coor, hon_x_coor, axis=1).reshape(-1,)
    ver_seg_coor = np.empty(shape=(disl_ver_seg.shape[1]*2,3))
    ver_seg_coor[:,0] = hon_x_coor
    #print(disl_ver_seg[0].reshape(-1,2))
    ver_seg_coor[:,1:3] = disl_ver_seg[0].reshape(-1,2)
    ref_coor = ver_seg_coor +0
    ref_coor[disl_ver_seg.shape[1]-1,1:3] = 0
    ref_coor[(disl_ver_seg.shape[1]-1)*2,1:3] = 0
    return ref_coor, ver_seg_coor
    
def image_coor_x(coor,seg_len):
    size_x = coor[-1,0] - coor[0,0] + seg_len
    coor1 = coor +0
    coor1[:,0] -= size_x
    coor2 = coor +0
    coor2[:,0] += size_x
    image_coor = np.append(coor1, coor2, axis=0)
    return image_coor

def pbc_coor_x(coor, seg_len, size_x = 0):
    if size_x == 0:
        size_x = coor[-1,0] - coor[0,0] + seg_len
    coor1 = coor +0
    coor1[:,0] -= size_x
    coor2 = coor +0
    coor2[:,0] += size_x
    pbc_coor = np.append(coor1, coor, axis=0)
    pbc_coor = np.append(pbc_coor, coor2, axis=0)
    return pbc_coor, size_x


class elastic_interaction_energy:
    def __init__(self, disl_hon_seg, disl_ver_seg, latt_a, latt_c, C13, C44, seg_len, b, rc):
        self.disl_hon_seg = disl_hon_seg
        self.disl_ver_seg = disl_ver_seg
        self.latt_a = latt_a
        self.latt_c = latt_c
        self.mu = C44
        self.nu = C13/(2*(C13+C44))
        self.seg_len = seg_len
        self.b = b
        self.rc = rc

    def seg_coor_image(self, state='initial'):
        if state == 'initial':
            # obtain (x,y,z) coordinate of hon part
            hon_seg_coor = switch_hon_seg_coor(self.disl_hon_seg, self.seg_len)
            hon_seg_coor_start = hon_seg_coor + 0
            hon_seg_coor_start[:,0] -= self.seg_len

            # image
            hon_seg_coor_image = image_coor_x(hon_seg_coor, self.seg_len)
            hon_coor_image_start_point = hon_seg_coor_image +0
            hon_coor_image_start_point[:,0] -= self.seg_len
            self.hon_coor_image_end_point = hon_seg_coor_image
            self.hon_coor_image_start_point = hon_coor_image_start_point

            # ver part
            ref_ver_coor, ver_seg_coor =switch_init_ver_seg_coor(self.disl_ver_seg, self.seg_len)
            ver_seg_coor_image = image_coor_x(ver_seg_coor, self.seg_len)
            ref_ver_coor_image = image_coor_x(ref_ver_coor, self.seg_len)
            self.ver_coor_image_end_point = ver_seg_coor_image
            self.ver_coor_image_start_point = ref_ver_coor_image
            
            # get all segment in one array
            self.start_point_image = np.append(self.hon_coor_image_start_point, self.ver_coor_image_start_point, axis=0)
            self.end_point_image   = np.append(self.hon_coor_image_end_point,   self.ver_coor_image_end_point,   axis=0)
            self.start_point = np.append(hon_seg_coor_start,  ref_ver_coor, axis=0)
            self.end_point   = np.append(hon_seg_coor, ver_seg_coor, axis=0)
            #print(self.end_point)
            #print(self.start_point)
    
    def calc(self, pbc=True):
        if pbc == True:
            self.seg_coor_image()
        W = 0
        #print(self.start_point,self.start_point_image )
        #pool = ml.Pool()
        for i in range(self.start_point.shape[0]):
            for j in range(self.start_point_image.shape[0]):
                x1 = self.start_point[i]
                x2 = self.end_point[i]
                x3 = self.start_point_image[j]
                x4 = self.end_point_image[j]
                # the segment exist (not zero length junction)
                if np.linalg.norm(x2-x1) >2 and np.linalg.norm(x4-x3) >2:
                    W_el = elastic_interaction_energy_ij(x1,x2,x3,x4,self.b,self.mu,self.nu,self.rc).calc_elastic_interaction()
                    W += W_el
                    #init = elastic_interaction_energy_ij(x1,x2,x3,x4,self.b,self.mu,self.nu,self.rc)
                    #W_el = pool.apply_async(init.calc_elastic_interaction,)
                    #W += W_el.get()
            for j in range(self.start_point.shape[0]):
                x1 = self.start_point[i]
                x2 = self.end_point[i]
                x3 = self.start_point[j]
                x4 = self.end_point[j]
                # the segment exist (not zero length junction)
                if np.linalg.norm(x2-x1) >2 and np.linalg.norm(x4-x3) >2:
                    W_el = elastic_interaction_energy_ij(x1,x2,x3,x4,self.b,self.mu,self.nu,self.rc).calc_elastic_interaction()
                    W += W_el/2
                    #init = elastic_interaction_energy_ij(x1,x2,x3,x4,self.b,self.mu,self.nu,self.rc)
                    #W_el = pool.apply_async(init.calc_elastic_interaction,)
                    #W += W_el.get()/2
        self.W_el = W
        return W


if __name__ == '__main__':
    for i in range(1,2):
        disl_hon_seg, disl_ver_seg = generate_dislocation_line(20,P=i)
        print(disl_hon_seg, disl_ver_seg)
        a = 2.9365976437421808
        c = 4.6410202908664067
        print('B len = ', np.sqrt(3)*a)
        print('Pi len = ', np.sqrt((np.sqrt(3)*a)**2+c**2) )
        print('P len = ', c)
        C13 = 83.235
        C44 = 54.91 #GPa
        seg_len = 2*a
        b = np.array([a,0,0])
        rc = 10*b[0]
        init = elastic_interaction_energy(disl_hon_seg, disl_ver_seg,a,c,C13,C44,seg_len,b,rc)
        init.seg_coor_image()
        init.calc()
        print(init.W_el)
        
