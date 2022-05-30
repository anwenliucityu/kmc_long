import numpy as np

class potential_info:
    def __init__(self,T):
        if T == 0:
            self.latt_const = {'a': 2.9365976437421808, 'c': 4.6410202908664067}
            self.C13 = 83.235
            self.C44 = 54.91
        elif T == 50:
            self.latt_const = {'a': 2.93589832714826, 'c': 4.65206265090668}
            self.C13 = 76.683237329627
            self.C44 = 57.5823577578576
        elif T == 100:
            self.latt_const = {'a': 2.93685760422465, 'c': 4.65819046226185}
            self.C13 = 72.553724977629
            self.C44 = 58.894072230488
        elif T == 150:
            self.latt_const = {'a': 2.93824479211728, 'c':4.66269844512587 }
            self.C13 = 69.2186417189646
            self.C44 = 58.6513357734666
        elif T == 200:
            self.latt_const = {'a': 2.9398882900745, 'c': 4.66639208401496} 
            self.C13 = 66.5484182302669
            self.C44 = 58.2546795794299
        elif T == 250:
            self.latt_const = {'a': 2.94149495512421 , 'c': 4.66980493976011} 
            self.C13 = 64.7146110607989
            self.C44 = 57.8128811978421
        elif T == 300:
            self.latt_const = {'a': 2.94307276759734 , 'c':4.67288606774279 }
            self.C13 = 62.1215150390139
            self.C44 = 56.6895104716219
        elif T == 350:
            self.latt_const = {'a': 2.94459149291013, 'c':4.67573480284092 }
            self.C13 = 60.8272693903142
            self.C44 = 56.0116637473055
        elif T == 400: 
            self.latt_const = {'a': 2.94608651296502, 'c': 4.67861099647893} 
            self.C13 = 60.1617900262731
            self.C44 = 54.7911849844006
        elif T == 450: 
            self.latt_const = {'a':2.94751911087698 , 'c': 4.68115879834816}
            self.C13 = 59.8722600091458
            self.C44 = 54.0845150380664
        elif T == 500:
            self.latt_const = {'a': 2.94905103124811, 'c': 4.68373656972749}
            self.C13 = 60.0974223000687
            self.C44 = 53.5867320991726
        elif T == 550:
            self.latt_const = {'a': 2.95036967772486, 'c': 4.68586723218424}
            self.C13 = 60.8766300147295
            self.C44 = 51.8086625261167
        elif T == 600:
            self.latt_const = {'a': 2.95180773265858, 'c': 4.68822150521338}
            self.C13 = 58.3465275928961
            self.C44 = 50.0506232687952
        elif T == 650:
            self.latt_const = {'a': 2.95296195379178, 'c':4.69027805888201 }
            self.C13 = 59.3295289698496
            self.C44 = 50.5876432463846
        elif T == 700:
            self.latt_const = {'a': 2.95436373943601, 'c': 4.69255637570191}
            self.C13 = 62.041873451027
            self.C44 = 46.8363650617183
        elif T == 750:
            self.latt_const = {'a': 2.95544778256945 , 'c': 4.69478836082782}
            self.C13 = 61.3057277469733
            self.C44 = 46.5512097205221
        elif T == 800:
            self.latt_const = {'a': 2.95686503143797, 'c': 4.69648088156127}
            self.C13 = 61.1594249587421
            self.C44 = 46.2494753895938
        elif T == 850:
            self.latt_const = {'a': 2.95797195263539, 'c': 4.69868866412223}
            self.C13 = 60.0821635858494
            self.C44 = 44.7488206350276
        elif T == 900:
            self.latt_const = {'a': 2.9592826951213, 'c': 4.70085160174947}
            self.C13 = 64.3583930371445
            self.C44 = 42.4102835883855
        


class core_probability:
    def __init__(self,T):
        self.a = -2.953840112370678e-07
        self.b = 0.00039435364900775277
        self.c = 0.5532402289918718
        self.aP = 9.151688960034703e-08
        self.bP = -0.00038421494906429744
        self.cP = 0.46430834363346063
        self.d = 2.915697990621726
        self.e = 2356.338488794463
        self.f = -7.907399258987868
        self.T = T

    def Pi_P_core(self, T, a, b, c):
        prob = a*T**2+b*T+c
        return prob

    def B_core(self,T, d, e, f):
        prob = d/(np.exp(e/T)-f)
        return prob

    def calc(self):
        if self.T != 0:
            B_core_prob = self.B_core(self.T, self.d, self.e, self.f)
        else:
            B_core_prob = 0
        i_core_prob = self.Pi_P_core(self.T, self.a, self.b, self.c)
        P_core_prob = self.Pi_P_core(self.T, self.aP, self.bP, self.cP)
        return B_core_prob, i_core_prob, P_core_prob

def basic_info(latt_const):
    a = latt_const['a']
    c = latt_const['c']
    b = np.array([a,0,0])
    rc = 6*a
    seg_len = 2*a
    unit_plane    = [[0,  np.sqrt(3)*a/2, 0],   #B
                     [0,  np.sqrt(3)*a/2, c],   #Pi1
                     [0, -np.sqrt(3)*a/2, c],   #Pi2
                     [0,               0, c]]   #P
    return b, rc, seg_len, np.array(unit_plane)


