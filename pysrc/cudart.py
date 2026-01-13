import numpy as np
import matplotlib.pyplot as plt

def set_plot_defaults(use_tex = True):
    """
    assign default plot settings before figure creation
    """
    ## FIGURE

    # test for tex environment
    if use_tex:
        plt.rcParams["text.usetex"] = "True"
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        # FONT
        plt.rcParams['font.serif']=['cm']
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']=['cm']
    else:
        plt.rcParams["text.usetex"] = "False"

    plt.rcParams['font.size']=8 # defval 18
    plt.rcParams['xtick.labelsize']=8
    plt.rcParams['ytick.labelsize']=8
    plt.rcParams['legend.fontsize']=8
    plt.rcParams['axes.titlesize']=8
    plt.rcParams['axes.labelsize']=8
    plt.rcParams['axes.linewidth']=1.5
    plt.rcParams["lines.linewidth"] = 2.2
    ## TICKS
    plt.rcParams['xtick.top']='False'
    plt.rcParams['xtick.bottom']='True'
    plt.rcParams['xtick.minor.visible']='True'
    plt.rcParams['xtick.direction']='out'
    plt.rcParams['ytick.left']='True'
    plt.rcParams['ytick.right']='True'
    plt.rcParams['ytick.minor.visible']='True'
    plt.rcParams['ytick.direction']='out'
    plt.rcParams['xtick.major.width']=1.5
    plt.rcParams['xtick.minor.width']=1
    plt.rcParams['xtick.major.size']=3
    plt.rcParams['xtick.minor.size']=9/4
    plt.rcParams['ytick.major.width']=1.5
    plt.rcParams['ytick.minor.width']=1
    plt.rcParams['ytick.major.size']=3
    plt.rcParams['ytick.minor.size']=9/4

def gen_npy(save_str):

    data = np.array([3.0,4.0,5.0,6.0], dtype=np.float32)

    np.save(save_str, data)

def gen_simple_data(npy_str, png_str):

    mb_dims = np.array([9,9,9])

    x_fc = np.linspace(-0.5, 0.5, mb_dims[0]+1)
    y_fc = np.linspace(-0.5, 0.5,  mb_dims[1]+1)
    z_fc = np.linspace(-0.5, 0.5,  mb_dims[2]+1)

    x_cc = 0.5 * (x_fc[1:] + x_fc[:-1])
    y_cc = 0.5 * (y_fc[1:] + y_fc[:-1])
    z_cc = 0.5 * (z_fc[1:] + z_fc[:-1])
    xx, yy, zz = np.meshgrid(x_cc, y_cc, z_cc, indexing="ij")
    data = (yy * zz).astype(np.float32)
    np.save(npy_str, data)

    set_plot_defaults()
    fig = plt.figure(figsize=(10.0 / 3, 10.0 / 3))
    ax = fig.add_subplot()

    yy, zz = np.meshgrid(y_fc, z_fc,indexing="ij")
    ax.pcolormesh(yy, zz, data[0,:,:])

    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(png_str, dpi=300, bbox_inches="tight")
    plt.close("all")

def show_npy(load_str, save_str):

    data = np.load(load_str)
    print(data)
    print(np.max(data))
    print(np.min(data))

    mb_dims = np.array([9, 9, 9])
    x_fc = np.linspace(-0.5, 0.5, mb_dims[0] + 1)
    y_fc = np.linspace(-0.5, 0.5, mb_dims[1] + 1)
    z_fc = np.linspace(-0.5, 0.5, mb_dims[2] + 1)
    yy, zz = np.meshgrid(y_fc, z_fc, indexing="ij")

    set_plot_defaults()
    fig = plt.figure(figsize=(10.0/3, 10.0/3))
    ax = fig.add_subplot()
    ax.pcolormesh(yy, zz, data, shading="flat")

    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(save_str, dpi=300, bbox_inches="tight")
    plt.close("all")

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
    np.save(save_str, emm.astype(np.float32))

def plot_sn_data(load_str, save_str):

    img = np.load(load_str)
    img_shape = np.shape(img)
    X = np.linspace(0,1,img_shape[0]+1)
    Y = np.linspace(0,1,img_shape[0]+1)
    XX, YY = np.meshgrid(X, Y, indexing="ij")

    cmap = "Greys"

    set_plot_defaults()
    fig = plt.figure(figsize=(10.0 / 3, 10.0 / 3))
    ax = fig.add_subplot()
    ax.pcolormesh(XX, YY, np.log10(img), vmin = -13, vmax=-10, cmap=cmap, shading="flat")

    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(save_str, dpi=300, bbox_inches="tight")
    plt.close("all")

if __name__ == "__main__":

    #gen_simple_data("../simdata/simple.npy", "../outputs/simple_data.png")
    #show_npy("../outputs/simple_img.npy", "../outputs/simple_img.png")
    #prep_sn_data("../simdata/interpolated_frame_gamm7_early_287.npy", "../simdata/sn.npy")
    plot_sn_data("../outputs/sn_imgs000.npy", "../outputs/sn_imgs000.png")
    plot_sn_data("../outputs/sn_imgs001.npy", "../outputs/sn_imgs001.png")