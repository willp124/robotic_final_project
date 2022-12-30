import math
import numpy as np
r = 0.339
c = np.array([[723*r, 1142*r, 305*r],
     [264*r, 681*r, 682*r],
     [1, 1, 1]])
arm = np.array([[300, 300, 500],
                [300, 500, 300],
                [1, 1, 1]])
c_inverse = np.linalg.inv(c)
t = np.dot(arm, c_inverse)
print(t)

coordinate = np.array([])
