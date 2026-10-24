import numpy as np
import math

# --- Constants of our robot ---
DH_TABLE = np.array([[math.pi/2, 0, 0, 0], # twist, torsion
                     [0, 93, 93, 50],       # alfa
                     [50, 0, 0, 0],         # d, link offset
                     [0, math.pi/2, 0, 0],  # offsets
                    ]) 

H45 = np.array([[1, 0, 0, -15],
                      [0, 1, 0,  45],
                      [0, 0, 1,   0],
                      [0, 0, 0,   1],
                    ], dtype=float)

def dh_matrix(t, r, a, d):
    dh = np.array([[np.cos(t), -np.sin(t)*np.cos(r),  np.sin(t)*np.sin(r), a*np.cos(t)],
                   [np.sin(t),  np.cos(t)*np.cos(r), -np.cos(t)*np.sin(r), a*np.sin(t)],
                   [0,          np.sin(r),            np.cos(r),           d          ],
                   [0,          0,                    0,                   1          ],
                ])
    return dh

def direct_kinematics(q, frame=4): # frame = 4 means that we are looking for the direct in joint 4, if frame = 5 means we look for direct in camera

    A1 = dh_matrix(q[0] + DH_TABLE[3][0], DH_TABLE[0][0], DH_TABLE[1][0], DH_TABLE[2][0])
    A2 = dh_matrix(q[1] + DH_TABLE[3][1], DH_TABLE[0][1], DH_TABLE[1][1], DH_TABLE[2][1])
    A3 = dh_matrix(q[2] + DH_TABLE[3][2], DH_TABLE[0][2], DH_TABLE[1][2], DH_TABLE[2][2])
    A4 = dh_matrix(q[3] + DH_TABLE[3][3], DH_TABLE[0][3], DH_TABLE[1][3], DH_TABLE[2][3])

    if frame == 1:
        return A1
    if frame == 2:
        return A1 @ A2
    if frame == 3:
        return A1 @ A2 @ A3
    if frame == 4:
        return A1 @ A2 @ A3 @ A4
    if frame == 5:
        return A1 @ A2 @ A3 @ A4 @ H45
    else:
        raise ValueError

def inverse_kinematics(x, x4z):
    q = np.zeros(4)
    
    theta1 = np.arctan2(x[1],x[0])
    q[0] = theta1

    h = np.sqrt(1 - x4z**2)

    o3 = np.array([x[0] - DH_TABLE[1][3] * np.cos(q[0]) * h, 
                   x[1] - DH_TABLE[1][3] * np.sin(q[0]) * h, 
                   x[2] - DH_TABLE[1][3] * x4z
                   ])

    r = np.sqrt(o3[0]**2 + o3[1]**2)
    s = o3[2] - DH_TABLE[2][0]

    cos3 = (r**2 + s**2 - DH_TABLE[1][1]**2 - DH_TABLE[1][2]**2) / (2*DH_TABLE[1][1]*DH_TABLE[1][2])
    sin3 = -1 * np.sqrt(1- cos3**2)
    theta3 = np.arctan2(sin3, cos3)
    q[2] = theta3

    theta2 = np.arctan2(s, r) - np.arctan2(DH_TABLE[1][2] * sin3, DH_TABLE[1][1] + DH_TABLE[1][2] *cos3)
    q[1]  = theta2 - DH_TABLE[3][1]

    q[3] = -np.arccos(x4z) - q[1] - q[2]
    return q

# For testing that it works
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print(direct_kinematics([0.3, -0.5, 0.8, 0.2]))

    print(inverse_kinematics([0, 0, 286, 0], 1))