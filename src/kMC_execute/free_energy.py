# calculate the core energy in a long segment

# input: the dislocation hon_seg and ver_seg 
# output: the free energy of the dislocation

import numpy as np
from md_input import junction_parallel_energy as E_jp
from md_input import basic_info, node_kink_energy
from line_math import vec_is_parallel

def get_start_ver_seg(ver_seg):
    shape = ver_seg.shape
    a = ver_seg.reshape(shape[0]*2,-1)
    a = np.r_[a[-1].reshape(1,-1),a[:]][:-1].reshape(shape)
    start_ver_seg = a
    return start_ver_seg

def check_kink_overlap(start_ver_seg, ver_seg):
    kink_vec =  ver_seg - start_ver_seg
    for i in range(kink_vec.shape[0]):
        if np.sum(kink_vec[i,:]) == 0: 
            start_ver_seg[i,:] = 0
            ver_seg[i,:] = 0
    return start_ver_seg, ver_seg

def core_3_kink_4_energy(core_3, kink_4, unit_plane):
    kink1 = [core_3[0],core_3[1],np.insert(kink_4[0][0],0,values=0)]
    kink2 = [core_3[0],core_3[1],np.insert(kink_4[0][1],0,values=0)]
    kink3 = [core_3[1],core_3[2],np.insert(kink_4[1][0],0,values=0)]
    kink4 = [core_3[1],core_3[2],np.insert(kink_4[1][1],0,values=0)]
    kinks = [kink1, kink2, kink3, kink4]

    F = 0
    for i in range(0,4,2):
        # check if two kink are the same type:
        if np.all(kinks[i][2]==kinks[i+1][2]) == True:
            # if kink are the same, we only have node energy
            F+=E_jp[str(int(kinks[i][0]))+str(int(kinks[i][1]))]
        else:
            # if one kink is 0 and another is not 0
            if np.all(kinks[i][2]==np.array([0,0,0])) == True or \
                np.all(kinks[i+1][2]==np.array([0,0,0])) == True:
                core1 = kinks[i][0]
                core2 = kinks[i+1][0]
                for j in range(4): # check if kink parallel to specific plane
                    if vec_is_parallel(kinks[i][2], unit_plane[j]) == True \
                        or vec_is_parallel(kinks[i+1][2], unit_plane[j]) == True:
                        kink_type=j
                        break
                F+=node_kink_energy(core1, core2, kink_type)
            else: # 2 kinks and they are not parallel:
                core1 = kinks[i][0]
                core2 = kinks[i+1][0]
                for k in range(2): # calculate 2 kink
                    for j in range(4): # check if kink parallel to specific plane
                        if vec_is_parallel(kinks[i+k][2], unit_plane[j]) == True:
                            kink_type=j
                            break
                    F+=node_kink_energy(core1, core2, kink_type)
    return F




#def kink_4_energy()

class seg_update_and_free_energy_difference:
    def __init__(self, hon_seg, ver_seg, a, loop, unit_plane):
        self.hon_seg = hon_seg
        self.ver_seg = ver_seg
        self.unit_plane = unit_plane
        self.a = a
        seg_len = self.a*2
        kink_pos = int(np.min(loop[:,0])/seg_len)
        self.pos = kink_pos # in the hon seg form
        self.loop=loop
        #print(self.pos)

    def update_segment_list(self):
        # return new hon_seg and new ver_seg with loop included:
        self.hon_seg[1:3,self.pos] = self.loop[1,1:3]
        self.ver_seg[(self.pos-1)*2+1,1,:] = self.loop[1,1:3]
        start_ver_seg = get_start_ver_seg(self.ver_seg)
        self.start_ver_seg, self.ver_seg = check_kink_overlap(start_ver_seg, self.ver_seg)
        #print(self.ver_seg- self.start_ver_seg)
     
    def calc(self):
        #self.update_segment_list()
        # calculate free energy before introduce the loop
        # we only consider 4 kinks here to reduce the works in calculation
        cores_3 =  [self.hon_seg[0][self.pos-1], self.hon_seg[0][self.pos], self.hon_seg[0][self.pos+1]]
        start_ver_seg = get_start_ver_seg(self.ver_seg)
        kinks = self.ver_seg - start_ver_seg
        kinks_4 = [kinks[self.pos-1],kinks[self.pos]]
        #calculate free energy at start state
        F0 = core_3_kink_4_energy(cores_3, kinks_4, self.unit_plane)

        #calculate free energy at end state
        self.update_segment_list() # update list
        kinks = self.ver_seg - start_ver_seg
        kinks_4 = [kinks[self.pos-1],kinks[self.pos]]
        #calculate free energy at start state
        F1 = core_3_kink_4_energy(cores_3, kinks_4, self.unit_plane)
        return F0, F1


if __name__ == '__main__':
    # input
    import generate_calc_dislocation_line as gcdl
    disl_hon_seg, disl_ver_seg = gcdl.generate_dislocation_line(10,P=1)
    #print(disl_hon_seg)#, disl_ver_seg)
    latt_const = {'a':2.9365976437421808, 'c':4.6410202908664067}
    a = 2.9365976437421808
    c = 4.6410202908664067
    C13 = 83.235
    C44 = 54.91 #GPa

    b, rc, seg_len, unit_plane = basic_info(latt_const)

    x1 = np.array([2*a,0,0])
    x2 = np.array([2*a,0,c])
    x3 = np.array([4*a,0,c])
    x4 = np.array([4*a,0,0])
    loop = np.array([x1,x2,x3,x4])

    # calculation
    F0,F1 = seg_update_and_free_energy_difference(disl_hon_seg, disl_ver_seg,a,loop, unit_plane).calc()
    print(F0,F1)