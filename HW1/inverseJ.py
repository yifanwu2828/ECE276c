import numpy as np
import scipy.linalg as linalg
from icecream import ic

if __name__ == "__main__":
    J = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    u, s, vh = np.linalg.svd(J)
    ic(J)
    ic(u, s, vh)
    
    J_inv, rank = linalg.pinv(J, return_rank=True)
    
    ic(J_inv, rank)
    