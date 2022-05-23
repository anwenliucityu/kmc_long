import numpy as np

def is_parallel(x1,x2,x3,x4):
    vec1 = x2-x1
    vec2 = x4-x3
    if abs(vec1[0]*vec2[1]-vec1[1]*vec2[0] + vec1[0]*vec2[2]-vec1[2]*vec2[0] + vec1[1]*vec2[2]-vec1[2]*vec2[1]) <1e-4:
        return True                                                                                                     
    else:
        return Fals

def is_perpendicular(x1,x2,x3,x4):
    vec1 = x2-x1
    vec2 = x4-x3
    if abs(vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2])<1e-4:
        return True
    else:
        return False

if __name__ == '__main__':
    x1 = np.array([0,0,0])
    x2 = np.array([1,0,0])
    x3 = np.array([10,0,0])
    x4 = np.array([10,1,0])
    result = is_perpendicular(x1,x2,x3,x4)
    print(result)
