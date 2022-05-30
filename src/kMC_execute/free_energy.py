# calculate the core energy in a long segment

# input: the dislocation hon_seg and ver_seg 
# output: the free energy of the dislocation

import numpy as np

def get_start_ver_seg(ver_seg):
    shape = ver_seg.shape
    a = ver_seg.reshape(shape[0]*shape[1]*2,-1)
    a = np.r_[a[-1].reshape(1,-1),a[:]][:-1].reshape(shape)
    start_ver_seg = a
    return start_ver_seg

def check_kink_overlap(start_ver_seg, ver_seg):
    kink_vec =  ver_seg - start_ver_seg
    print(kink_vec.shape)
    for i in range(kink_vec.shape[1]):
        if np.sum(kink_vec[-1,i,:]) == 0:
            start_ver_seg[-1,i,:] = 0
            ver_seg[-1,i,:] = 0
    return start_ver_seg, ver_seg

class free_energy_difference:
    def __init__(self, hon_seg, ver_seg, a, loop):
        self.hon_seg = hon_seg
        self.ver_seg = ver_seg
        self.a = a
        seg_len = self.a*2
        kink_pos = int(np.min(loop[:,0])/seg_len)
        self.pos = kink_pos # in the hon seg form
        #print(self.pos)

    def update_segment_list(self):
        # return new hon_seg and new ver_seg with loop included:
        self.hon_seg[1:3,self.pos] = self.loop[1,1:3]
        self.ver_seg[0,(self.pos-1)*2+1,1,:] = self.loop[1,1:3]
        start_ver_seg = get_start_ver_seg(self.ver_seg)
        self.start_ver_seg, self.ver_seg = check_kink_overlap(start_ver_seg, self.ver_seg)
     
    def calc(self, free_energy_change=True):
        self.update_segment_list()
        if free_energy_change==True:
            #计算free energy
            True
        else:
            return 0
        # calculate the free energy

if __name__ == '__main__':
    # input
    import generate_calc_dislocation_line as gcdl
    disl_hon_seg, disl_ver_seg = gcdl.generate_dislocation_line(10,P=1)
    #print(disl_hon_seg)#, disl_ver_seg)
    a = 2.9365976437421808
    c = 4.6410202908664067
    C13 = 83.235
    C44 = 54.91 #GPa
    seg_len = 2*a
    b = np.array([a,0,0])
    rc = 6*b[0]

    x1 = np.array([0*a,0,0])
    x2 = np.array([0*a,0,c])
    x3 = np.array([2*a,0,c])
    x4 = np.array([2*a,0,0])
    loop = np.array([x1,x2,x3,x4])

    # calculation
    F = free_energy_difference(disl_hon_seg, disl_ver_seg,a,loop).calc(loop=loop)
    print(F)