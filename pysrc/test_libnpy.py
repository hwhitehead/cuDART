import numpy as np

def gen_npy(save_str):

    data = np.array([3,4,5,6])

    np.save(save_str, data)

if __name__ == "__main__":

    gen_npy("../simdata/data.npy")
