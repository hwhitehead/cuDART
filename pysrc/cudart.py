import numpy as np

def gen_npy(save_str):

    data = np.array([3.0,4.0,5.0,6.0], dtype=np.float32)

    np.save(save_str, data)

def prep_sn_data(load_str, save_str):

    data = np.load(load_str).astype(dtype=np.float32)
    quart_emm = np.einsum("kji->ijk", data)
    quart_dims = np.shape(quart_emm)
    hdim = quart_dims[0]
    # populate FULL array (x4 size)
    emm = np.zeros(shape=(2 * quart_dims[0], 2 * quart_dims[1], quart_dims[2]))
    emm[hdim:, hdim:, :] = quart_emm  # +x, +y quadrant
    emm[hdim:, :hdim, :] = quart_emm[:, ::-1, :]  # +x, -y quadrant
    emm[:hdim, hdim:, :] = quart_emm[::-1, :, :]  # -x, +y quadrant
    emm[:hdim, :hdim, :] = quart_emm[::-1, ::-1, :]  # -x, -y quadrant

    np.save(save_str, data)

if __name__ == "__main__":

    #gen_npy("../simdata/data.npy")
    prep_sn_data("../simdata/interpolated_frame_gamm7_early_287.npy", "../simdata/sn_low.npy")