
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_z2(f, fignum=None, range=None, color_map='gray'):

    # plt.figure(fignum)

    if range is None:
        plt.imshow(f, interpolation='nearest', cmap=cm.get_cmap(color_map))
    else:
        plt.imshow(f, interpolation='nearest', cmap=cm.get_cmap(color_map),
                   vmin=range[0], vmax=range[1])

    plt.xticks(np.arange(f.shape[1]), [str(i) for i in np.arange(f.shape[1])])
    plt.yticks(np.arange(f.shape[0]), [str(i) for i in np.arange(f.shape[0])])
