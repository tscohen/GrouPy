import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from groupy.gfunc.plot import plot_z2


def plot_p4(f, fignum=None, rlabels='cayley', rcolor='red', rlinestyle='-',
            fontsize=20, labelpad_factor_1=1.5, labelpad_factor_2=1.5, figsize=(3, 3)):
    """
    Plot a function f : p4 -> R or f : p4 -> R^3.

    :param f: array of shape (4, nx, ny) or (4, nx, ny, 3) for a color plot.
    :param fignum: which figure the plot to.
    :param rlabels: the type of labels to use for the 4 patches.
    :param rcolor: the color of the rotation arrows.
    :param rlinestyle: the linestyle of the rotation arrows.
    :param fontsize: size of the font used to label the 4 patches.
    :param labelpad_factor_1: tweak the position of the label.
    :param labelpad_factor_2: tweak the position of the label.
    :param figsize: size of figure.
    """

    assert rlabels in ['radians', 'cayley', 'indices', 'none']
    assert f.shape[0] == 4
    assert f.ndim == 3 or f.ndim == 4
    ny, nx = f.shape[1:3]

    rlabel_names = {
        'radians': ['$0$', '$\\frac{\pi}{2}$', '$\\pi$', '$\\frac{3 \pi}{2}$'],
        'cayley': ['$e$', '$r$', '$r^2$', '$r^3$'],
        'indices': [0, 1, 2, 3],
        'none': ['', '', '', '']
    }

    fig = plt.figure(fignum, figsize=(2 * f.shape[1], 2 * f.shape[2]))
    fignum = fig.number
    main_ax = fig.gca()

    figtr = fig.transFigure.inverted()  # Display -> Figure

    ax_e = fig.add_subplot(3, 3, 2)
    plot_z2(f[0], fignum=fignum)
    ax_e.xaxis.set_label_position('bottom')
    ax_e.set_xlabel(rlabel_names[rlabels][0], fontsize=fontsize, labelpad=labelpad_factor_1 * fontsize)
    ax_e.set_xticks([])
    ax_e.set_yticks([])

    ax_r3 = fig.add_subplot(3, 3, 6)
    plot_z2(f[3], fignum=fignum)
    ax_r3.yaxis.set_label_position('left')
    ax_r3.set_ylabel(rlabel_names[rlabels][3], fontsize=fontsize, rotation='horizontal', va='center', labelpad=labelpad_factor_2 * fontsize)
    ax_r3.set_xticks([])
    ax_r3.set_yticks([])

    ax_r2 = fig.add_subplot(3, 3, 8)
    plot_z2(f[2], fignum=fignum)
    ax_r2.xaxis.set_label_position('top')
    ax_r2.set_xlabel(rlabel_names[rlabels][2], fontsize=fontsize, labelpad=labelpad_factor_1 * fontsize)
    ax_r2.set_xticks([])
    ax_r2.set_yticks([])

    ax_r = fig.add_subplot(3, 3, 4)
    plot_z2(f[1], fignum=fignum)
    ax_r.yaxis.set_label_position('right')
    ax_r.set_ylabel(rlabel_names[rlabels][1], fontsize=fontsize, rotation=0, va='center', labelpad=labelpad_factor_2 * fontsize)
    ax_r.set_xticks([])
    ax_r.set_yticks([])

    # Create pixel coordinate in the subplot coordinate systems for each beginning and enpoint of the arrows
    pt_right = (nx - 0.25, ny // 2)
    pt_top = (nx // 2, -0.75)
    pt_bottom = (nx // 2, ny - 0.25)
    pt_left = (-0.75, ny // 2)
    pt_center = (nx // 2, ny // 2)

    # Transform to figure coordinates
    pt_e_r = figtr.transform(ax_e.transData.transform(pt_left))
    pt_r_e = figtr.transform(ax_r.transData.transform(pt_top))

    pt_r_r2 = figtr.transform(ax_r.transData.transform(pt_bottom))
    pt_r2_r = figtr.transform(ax_r2.transData.transform(pt_left))

    pt_r2_r3 = figtr.transform(ax_r2.transData.transform(pt_right))
    pt_r3_r2 = figtr.transform(ax_r3.transData.transform(pt_bottom))

    pt_r3_e = figtr.transform(ax_r3.transData.transform(pt_top))
    pt_e_r3 = figtr.transform(ax_e.transData.transform(pt_right))

    arrow = FancyArrowPatch(
            pt_e_r,
            pt_r_e,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=10, angleB=-100',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_r_r2,
            pt_r2_r,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=100, angleB=170',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_r2_r3,
            pt_r3_r2,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=190, angleB=260',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_r3_e,
            pt_e_r3,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=280, angleB=-10',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    main_ax.axis('off')

    fig.set_size_inches(figsize, forward=True)
