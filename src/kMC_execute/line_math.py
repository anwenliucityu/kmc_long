import numpy as np
import math as m

def is_parallel(x1,x2,x3,x4):
    vec1 = x2-x1
    vec2 = x4-x3
    if abs(vec1[0]*vec2[1]-vec1[1]*vec2[0] + vec1[0]*vec2[2]-vec1[2]*vec2[0] + vec1[1]*vec2[2]-vec1[2]*vec2[1]) <1e-4 and np.any(vec1) != 0 and np.any(vec2) != 0:
        return True                 
    else:
        return False

def vec_is_parallel(vec1,vec2):
    # vec1 and vec2 should not be 0:
    if abs(vec1[0]*vec2[1]-vec1[1]*vec2[0] + vec1[0]*vec2[2]-vec1[2]*vec2[0] + vec1[1]*vec2[2]-vec1[2]*vec2[1]) <1e-4 and np.any(vec1) != 0 and np.any(vec2) != 0:
        return True               
    else:
        return False

def is_perpendicular(x1,x2,x3,x4):
    vec1 = x2-x1
    vec2 = x4-x3
    if abs(vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2])<1e-4:
        return True
    else:
        return False

# Rotate the coordinate [x,y,z] anti-clockwise, keep coor system still
def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
  return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
  return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

if __name__ == '__main__':
    x1 = np.array([0,0,0])
    x2 = np.array([0,0,1])
    x3 = np.array([0,0,0])
    x4 = np.array([1,0,0])
    #result = is_parallel(x1,x2,x3,x4)
    a = np.array([0,0,1])
    b = np.array([1,0,0])
    result = vec_is_parallel(a,b)
    print(result)
