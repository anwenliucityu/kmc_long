import numpy as np
import md_input as md
import generate_calc_dislocation_line as gcdl
import total_and_incremental_energy as taie


def main(T, seg_num=100):
    #--------- prework --------#

    # lattice constant and C13, C14 at temperature T
    potential_info = md.potential_info(T)
    latt_const = potential_info.latt_const
    C13 = potential_info.C13
    C44 = potential_info.C44

    # initiate basic parameter
    b, rc, seg_len, unit_plane = md.basic_info(latt_const)

    # obtain core probability at temperature T post-analysed from MD:
    B_core_prob, i_core_prob, P_core_prob = md.core_probability(T).calc()

    # generate a dislocation line.
    disl_hon_seg, disl_ver_seg = gcdl.generate_dislocation_line(seg_num, init_core=3, a=latt_const['a'],c=latt_const['c'])
    #print(disl_hon_seg, disl_ver_seg)

    # core energy
    core = taie.disl_core(disl_hon_seg, disl_ver_seg, B_core_prob, i_core_prob, P_core_prob)
    core_frequency = core.calc()
    print('core probability sum for initial configuration = ', core_frequency)

    # elastic interaction energy (not necessary)
    #pbc_config = gcdl.elastic_interaction_energy(disl_hon_seg, disl_ver_seg,latt_const['a'],latt_const['c'],C13,C44,seg_len,b,rc)
    #E_el_inter = pbc_config.calc()
    #print('elastic interaction energy for initial configuration = ', E_el_inter, ' eV')

    #-----------first step----------------#
    






if __name__ == '__main__':
    main(0,seg_num=20)

