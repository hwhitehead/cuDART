import numpy as np

def gen_npy(save_str):

    data = np.array([3.0,4.0,5.0,6.0], dtype=np.float32)

    np.save(save_str, data)

if __name__ == "__main__":

    gen_npy("../simdata/data.npy")
