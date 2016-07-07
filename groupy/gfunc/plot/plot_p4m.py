import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from groupy.gfunc.plot.plot_z2 import plot_z2


# Miniature plot:
# plot_p4m(imf.reshape(2, 4, 7, 7), rlabels='cayley2', fontsize=10,
#          labelpad_factor_1= .2, labelpad_factor_2=.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2, figsize=(2.5, 2.5)


def plot_p4m(f, fignum=None, rlabels='cayley_mr', rcolor='red', mcolor='blue', rlinestyle='-', mlinestyle='-',
             fontsize=20, labelpad_factor_1=1.5, labelpad_factor_2=1.5, labelpad_factor_3=2.5, labelpad_factor_4=2.5,
             figsize=(3, 3)):
    """
    Plot a function f : p4m -> R or f : p4m -> R^3.

    :param f: array of shape (2, 4, nx, ny) or (2, 4, nx, ny, 3) for a color plot.
    :param fignum: which figure the plot to.
    :param rlabels: the type of labels to use for the 8 patches.
    :param rcolor: the color of the rotation arrows.
    :param mcolor: the color of the mirror lines.
    :param rlinestyle: the linestyle of the rotation arrows.
    :param mlinestyle: the linestyle of the mirror lines.
    :param fontsize: size of the font used to label the 8 patches.
    :param labelpad_factor_1: tweak the position of the label.
    :param figsize: size of figure.
    """

    assert f.shape[0] == 2
    assert f.shape[1] == 4
    assert f.ndim == 4 or f.ndim == 5
    ny, nx = f.shape[2:4]

    rlabel_names = {
        'cayley_rm': ['$e$', '$r$', '$r^2$', '$r^3$', '$m$', '$r^3m$', '$r^2m$', '$rm$'],
        'cayley_mr': ['$e$', '$r$', '$r^2$', '$r^3$', '$m$', '$mr$', '$mr^2$', '$mr^3$'],
        'cayley2': ['$e$', '$r$', '$r^2$', '$r^3$', '$m$', '$mr$\n$=$\n$r^3m$', '$r^2m = mr^2$', '$mr^3$\n$=$\n$rm$'],
        'none': ['', '', '', '', '', '', '', '']
    }

    fig = plt.figure(fignum, figsize=(2 * f.shape[1], 2 * f.shape[2]))
    fignum = fig.number
    main_ax = fig.gca()

    # Inner ring
    ax_e = fig.add_subplot(5, 5, 8)
    plot_z2(f[0, 0], fignum=fignum)
    ax_e.xaxis.set_label_position('bottom')
    ax_e.set_xlabel(
        rlabel_names[rlabels][0],
        fontsize=fontsize,
        labelpad=labelpad_factor_1 * fontsize)
    ax_e.set_xticks([])
    ax_e.set_yticks([])

    ax_r = fig.add_subplot(5, 5, 12)
    plot_z2(f[0, 1], fignum=fignum)
    ax_r.yaxis.set_label_position('right')
    ax_r.set_ylabel(
        rlabel_names[rlabels][1],
        fontsize=fontsize,
        rotation='horizontal',
        va='center',
        labelpad=labelpad_factor_2 * fontsize)
    ax_r.set_xticks([])
    ax_r.set_yticks([])

    ax_r2 = fig.add_subplot(5, 5, 18)
    plot_z2(f[0, 2], fignum=fignum)
    ax_r2.xaxis.set_label_position('top')
    ax_r2.set_xlabel(
        rlabel_names[rlabels][2],
        fontsize=fontsize,
        labelpad=labelpad_factor_1 * fontsize)
    ax_r2.set_xticks([])
    ax_r2.set_yticks([])

    ax_r3 = fig.add_subplot(5, 5, 14)
    plot_z2(f[0, 3], fignum=fignum)
    ax_r3.yaxis.set_label_position('left')
    ax_r3.set_ylabel(
        rlabel_names[rlabels][3],
        fontsize=fontsize,
        rotation=0,
        va='center',
        labelpad=labelpad_factor_2 * fontsize)
    ax_r3.set_xticks([])
    ax_r3.set_yticks([])

    # Outer ring
    ax_m = fig.add_subplot(5, 5, 3)
    plot_z2(f[1, 0], fignum=fignum)
    ax_m.xaxis.set_label_position('top')
    ax_m.set_xlabel(
        rlabel_names[rlabels][4],
        fontsize=fontsize,
        labelpad=labelpad_factor_3 * fontsize)
    ax_m.set_xticks([])
    ax_m.set_yticks([])

    ax_mr3 = fig.add_subplot(5, 5, 11)
    plot_z2(f[1, 1], fignum=fignum)
    ax_mr3.yaxis.set_label_position('left')
    ax_mr3.set_ylabel(
        rlabel_names[rlabels][5],
        fontsize=fontsize,
        rotation='horizontal',
        va='center',
        labelpad=labelpad_factor_4 * fontsize)
    ax_mr3.set_xticks([])
    ax_mr3.set_yticks([])

    ax_mr2 = fig.add_subplot(5, 5, 23)
    plot_z2(f[1, 2], fignum=fignum)
    ax_mr2.xaxis.set_label_position('bottom')
    ax_mr2.set_xlabel(
        rlabel_names[rlabels][6],
        fontsize=fontsize,
        labelpad=labelpad_factor_3 * fontsize)
    ax_mr2.set_xticks([])
    ax_mr2.set_yticks([])

    ax_mr = fig.add_subplot(5, 5, 15)
    plot_z2(f[1, 3], fignum=fignum)
    ax_mr.yaxis.set_label_position('right')
    ax_mr.set_ylabel(
        rlabel_names[rlabels][7],
        fontsize=fontsize,
        rotation=0,
        va='center',
        labelpad=labelpad_factor_4 * fontsize)
    ax_mr.set_xticks([])
    ax_mr.set_yticks([])

    # Create pixel coordinate in the subplot coordinate systems for each beginning and enpoint of the arrows
    pt_right = (nx - 0.25, ny // 2)
    pt_top = (nx // 2, -0.75)
    pt_bottom = (nx // 2, ny - 0.25)
    pt_left = (-0.75, ny // 2)
    pt_center = (nx // 2, ny // 2)

    figtr = fig.transFigure.inverted()  # Display -> Figure

    # Transform to figure coordinates
    # Forward rotation arrows
    pt_e_r = figtr.transform(ax_e.transData.transform(pt_left))
    pt_r_e = figtr.transform(ax_r.transData.transform(pt_top))

    pt_r_r2 = figtr.transform(ax_r.transData.transform(pt_bottom))
    pt_r2_r = figtr.transform(ax_r2.transData.transform(pt_left))

    pt_r2_r3 = figtr.transform(ax_r2.transData.transform(pt_right))
    pt_r3_r2 = figtr.transform(ax_r3.transData.transform(pt_bottom))

    pt_r3_e = figtr.transform(ax_r3.transData.transform(pt_top))
    pt_e_r3 = figtr.transform(ax_e.transData.transform(pt_right))

    # Mirrored rotation arrows
    pt_m_mr = figtr.transform(ax_m.transData.transform(pt_right))
    pt_mr_m = figtr.transform(ax_mr.transData.transform(pt_top))

    pt_mr_mr2 = figtr.transform(ax_mr.transData.transform(pt_bottom))
    pt_mr2_mr = figtr.transform(ax_mr2.transData.transform(pt_right))

    pt_mr2_mr3 = figtr.transform(ax_mr2.transData.transform(pt_left))
    pt_mr3_mr2 = figtr.transform(ax_mr3.transData.transform(pt_bottom))

    pt_mr3_m = figtr.transform(ax_mr3.transData.transform(pt_top))
    pt_m_mr3 = figtr.transform(ax_m.transData.transform(pt_left))

    # Mirroring lines
    pt_e_m = figtr.transform(ax_e.transData.transform(pt_center))
    pt_m_e = figtr.transform(ax_m.transData.transform(pt_center))

    pt_r_mr3 = figtr.transform(ax_r.transData.transform(pt_center))
    pt_mr3_r = figtr.transform(ax_mr3.transData.transform(pt_center))

    pt_r2_mr2 = figtr.transform(ax_r2.transData.transform(pt_center))
    pt_mr2_r2 = figtr.transform(ax_mr2.transData.transform(pt_center))

    pt_r3_mr = figtr.transform(ax_r3.transData.transform(pt_center))
    pt_mr_r3 = figtr.transform(ax_mr.transData.transform(pt_center))

    # Draw rotation arrows
    arrow = FancyArrowPatch(
            pt_e_r,
            pt_r_e,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=10, angleB=-100',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle
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
            linestyle=rlinestyle
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
            linestyle=rlinestyle
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
            linestyle=rlinestyle
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_m_mr,
            pt_mr_m,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=170, angleB=280',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_mr_mr2,
            pt_mr2_mr,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=260, angleB=10',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_mr2_mr3,
            pt_mr3_mr2,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=-10, angleB=100',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_mr3_m,
            pt_m_mr3,
            transform=fig.transFigure,
            connectionstyle='angle3, angleA=260, angleB=10',
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle
    )
    fig.patches.append(arrow)

    # Draw mirror lines
    main_ax.add_line(Line2D((pt_e_m[0], pt_m_e[0]), (pt_e_m[1], pt_m_e[1]), zorder=0, linewidth=4, color=mcolor, transform=fig.transFigure, linestyle=mlinestyle))
    main_ax.add_line(Line2D((pt_r_mr3[0], pt_mr3_r[0]), (pt_r_mr3[1], pt_mr3_r[1]), zorder=0, linewidth=4, color=mcolor, transform=fig.transFigure, linestyle=mlinestyle))
    main_ax.add_line(Line2D((pt_r2_mr2[0], pt_mr2_r2[0]), (pt_r2_mr2[1], pt_mr2_r2[1]), zorder=0, linewidth=4, color=mcolor, transform=fig.transFigure, linestyle=mlinestyle))
    main_ax.add_line(Line2D((pt_r3_mr[0], pt_mr_r3[0]), (pt_r3_mr[1], pt_mr_r3[1]), zorder=0, linewidth=4, color=mcolor, transform=fig.transFigure, linestyle=mlinestyle))

    main_ax.axis('off')

    fig.set_size_inches(figsize, forward=True)
