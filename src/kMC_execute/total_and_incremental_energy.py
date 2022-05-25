import numpy as np
import generate_calc_dislocation_line as gcdl
import elastic_energy as ee
import sys
#sys.path.append('../kink_energy_fitting/node_2nd_interaction') read fitting data

class disl_core:
    def __init__(self, disl_hon_seg, disl_ver_seg, B_core_prob, i_core_prob, P_core_prob ):
        self.disl_hon_seg = disl_hon_seg
        self.disl_ver_seg = disl_ver_seg
        self.B_core_prob = B_core_prob
        self.i_core_prob = i_core_prob
        self.P_core_prob = P_core_prob

    def calc(self,):
        seg_core = self.disl_hon_seg[0,:]
        all_seg_core_prob = np.sum(seg_core==0)*self.B_core_prob + \
                        (np.sum(seg_core==1)+np.sum(seg_core==2))*self.i_core_prob +\
                            np.sum(seg_core==3)*self.P_core_prob
        self.core_frequency = all_seg_core_prob
        return self.core_frequency

#class kink_junc_energy:
 #   def __init__(self, ):


class loop_line_incremental_energy:
    def __init__(self, loop, disl_hon_seg, disl_ver_seg, latt_a, latt_c, C13, C44, seg_len, b, rc):
        # loop =  [x1,x2,x3,x4]
        self.end_loop =  np.array(loop)
        self.sta_loop = np.array([loop[3], loop[0], loop[1], loop[2]])

        self.disl_hon_seg = disl_hon_seg
        self.disl_ver_seg = disl_ver_seg
        self.latt_a = latt_a
        self.latt_c = latt_c
        self.mu = C44
        self.nu = C13/(2*(C13+C44))
        self.seg_len = seg_len
        self.b = b
        self.rc = rc

    def seg_coor_pbc(self, state='initial'):
        if state == 'initial':
            # obtain (x,y,z) coordinate of hon part
            hon_seg_coor = gcdl.switch_hon_seg_coor(self.disl_hon_seg, self.seg_len)
            hon_seg_coor_start = hon_seg_coor + 0
            hon_seg_coor_start[:,0] -= self.seg_len

            # pbc
            hon_seg_coor_pbc, size_x = gcdl.pbc_coor_x(hon_seg_coor, self.seg_len)
            hon_coor_pbc_start_point = hon_seg_coor_pbc +0
            hon_coor_pbc_start_point[:,0] -= self.seg_len
            self.hon_coor_pbc_end_point = hon_seg_coor_pbc
            self.hon_coor_pbc_start_point = hon_coor_pbc_start_point

            # ver part
            ref_ver_coor, ver_seg_coor    = gcdl.switch_init_ver_seg_coor(self.disl_ver_seg, self.seg_len)
            self.ver_sta_coor             = ref_ver_coor
            self.ver_end_coor             = ver_seg_coor

            ver_seg_coor_pbc, _           = gcdl.pbc_coor_x(ver_seg_coor, self.seg_len)
            ref_ver_coor_pbc, _           = gcdl.pbc_coor_x(ref_ver_coor, self.seg_len)
            self.ver_coor_pbc_end_point   = ver_seg_coor_pbc
            self.ver_coor_pbc_start_point = ref_ver_coor_pbc

            # line self coordinate
            self.start_point = np.append(hon_seg_coor_start,  ref_ver_coor, axis=0)
            self.end_point   = np.append(hon_seg_coor, ver_seg_coor, axis=0)

            # get pbc dislocation line coordinate in one array
            self.sta_line_pbc = np.append(self.hon_coor_pbc_start_point, self.ver_coor_pbc_start_point, axis=0)
            self.end_line_pbc   = np.append(self.hon_coor_pbc_end_point,   self.ver_coor_pbc_end_point,   axis=0)

            # pbc loop
            #self.end_loop_pbc = gcdl.pbc_coor_x(end_loop, self.seg_len, size_x = size_x)
            #self.sta_loop_pbc = gcdl.pbc_coor_x(sta_loop, self.seg_len, size_x = size_x)

    def calc_elastic_inter_energy(self, pbc=True):
        if pbc == True:
            self.seg_coor_pbc()
        W = 0
        for i in range(self.end_loop.shape[0]):
            for j in range(self.sta_line_pbc.shape[0]):
                x1 = self.sta_loop[i]
                x2 = self.end_loop[i]
                x3 = self.sta_line_pbc[j]
                x4 = self.end_line_pbc[j]
                # the segment exist (not zero length junction)
                if np.linalg.norm(x2-x1) >1 and np.linalg.norm(x4-x3) >1:
                    W_el = ee.elastic_interaction_energy_ij(x1,x2,x3,x4,self.b,self.mu,self.nu,self.rc).calc_elastic_interaction()
                    W += W_el
        self.W = W
        return W

    def loop_core_energy(self, unit_plane, w_dat):
        seg_core = self.disl_hon_seg[0,:]
        loop_vec = self.end_loop - self.sta_loop
        kink_vec = self.ver_end_coor - self.ver_sta_coor 

        loop_type=np.zeros(shape=(37,))
        # determine the kink type in the loop
        kink_plane = []
        for i in range(loop_vec.shape[0]):
            for j in range(4):
                if ee.vec_is_parallel(loop_vec[i], unit_plane[j]) == True:
                    #print(loop_vec[i], unit_plane[j])
                    kink_plane.append(j)
        assert kink_plane[0] == kink_plane[1]
        kink_plane = list(set(kink_plane))

        # determine the node part in the loop


if __name__ == '__main__':
    import md_input as md
    disl_hon_seg, disl_ver_seg = gcdl.generate_dislocation_line(20,P=1)
    #print(disl_hon_seg.shape, disl_ver_seg)
    a = 2.9365976437421808
    c = 4.6410202908664067
    C13 = 83.235
    C44 = 54.91 #GPa
    seg_len = 2*a
    b = np.array([a,0,0])
    rc = 10*b[0]

    x1 = np.array([5,0,0])
    x2 = np.array([0,0,0])
    x3 = np.array([0,0,5])
    x4 = np.array([5,0,5])
    loop = [x1,x2,x3,x4]
    
    T = 0
    potential_info = md.potential_info(T)
    latt_const = potential_info.latt_const
    b, rc, seg_len, unit_plane = md.basic_info(latt_const)


    init = loop_line_incremental_energy(loop, disl_hon_seg, disl_ver_seg, a, c, C13, C44, seg_len, b, rc)
    init.calc_elastic_inter_energy()
    print(init.W)
    print(unit_plane)
    init.loop_core_energy(unit_plane)










