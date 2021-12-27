"""
This file is part of the accompanying code to our paper
Jiang, S., Zheng, Y., Wang, C., & Babovic, V. (2021). Uncovering flooding mecha-
nisms across the contiguous United States through interpretive deep learning on
representative catchments. Water Resources Research, 57, e2021WR030185.
https://doi.org/10.1029/2021WR030185.

Copyright (c) 2021 Shijie Jiang. All rights reserved.

You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

def plot_peaks(Q, peak_dates, plot_range=[None, None], linecolor="tab:brown", markercolor="tab:red", figsize=(7.5, 2.0)):
    """
    Plot the identified flood peaks.

    Parameters
    ----------
    Q: pandas series of streamflow observations.
    peak_dates: a sequence of flood peaks' occurrence dates.
    plot_range: the date range of the plot, it can be a pair of date strings (default: [None, None]).
    linecolor: the color of the line (default: 'tab:brown').
    markercolor: the color of the marker (default: 'tab:red').
    figsize: the width and height of the figure in inches (default: (7.5, 2.0)).

    """

    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()

    plot_range[0] = Q.index[0] if plot_range[0] == None else plot_range[0]
    plot_range[1] = Q.index[-1] if plot_range[1] == None else plot_range[1]

    ax.plot(Q["flow"].loc[plot_range[0]:plot_range[1]], color=linecolor, lw=1.0)
    ax.plot(
        Q.loc[peak_dates, "flow"].loc[plot_range[0]:plot_range[1]],
        "*",
        c=markercolor,
        markersize=8,
    )

    ax.set_title(f"Identified flood peaks from {plot_range[0]} to {plot_range[1]}")
    ax.set_ylabel("flow(mm)")

    plt.show()


def plot_eg_individual(dataset, peak_eg_dict, peak_eg_var_dict, peak_date, title_suffix=None, linewidth=1.5, figsize=(10, 3)):

    eg_plot = dataset.loc[pd.date_range(end=peak_date, periods=list(peak_eg_dict.values())[0].shape[1]+1, freq='d')[:-1]]

    eg_plot.loc[:, "prcp_eg"] = abs(peak_eg_dict[pd.to_datetime(peak_date)][0, :, 0])
    eg_plot.loc[:, "temp_eg"] = abs(peak_eg_dict[pd.to_datetime(peak_date)][0, :, 1])
    eg_plot.loc[:, "prcp_eg_val"] = abs(peak_eg_var_dict[pd.to_datetime(peak_date)][0, :, 0])
    eg_plot.loc[:, "temp_eg_val"] = abs(peak_eg_var_dict[pd.to_datetime(peak_date)][0, :, 1])

    fig = plt.figure(constrained_layout=False, figsize=figsize)

    gs1 = fig.add_gridspec(nrows=2, ncols=1, hspace=0, left=0.00, right=0.45, height_ratios=[2.5, 1.5])
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[1, 0])

    gs2 = fig.add_gridspec(nrows=2, ncols=1, hspace=0, left=0.55, right=1.00, height_ratios=[2.5, 1.5])
    ax3 = fig.add_subplot(gs2[0, 0])
    ax4 = fig.add_subplot(gs2[1, 0])

    for ax in [ax1, ax3]:
        ax.spines["bottom"].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

    for ax in [ax2, ax4]:
        ax.set_ylabel(r'$\phi^{EG}_{i}$')
        ax.spines["top"].set_visible(False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylim(bottom=np.min(peak_eg_dict[pd.to_datetime(peak_date)]),
                 top=np.max(peak_eg_dict[pd.to_datetime(peak_date)]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))


    ax1.plot(eg_plot['prcp'], color='k', lw=linewidth)
    ax1.set_ylabel('P [mm]', ha='center', y=0.5)

    ax2.plot(eg_plot['prcp_eg'], color='blue', lw=linewidth)
    ax2.fill_between(eg_plot['prcp_eg'].index,
                     eg_plot['prcp_eg']-eg_plot.loc[:, "prcp_eg_val"],
                     eg_plot['prcp_eg']+eg_plot.loc[:, "prcp_eg_val"], color='blue', alpha=0.3)
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='blue')


    ax3.plot(eg_plot['tmean'], color='k', lw=linewidth)
    ax3.set_ylabel('T [\u2103]', ha='center', y=0.5)

    ax4.plot(eg_plot['temp_eg'], color='red', lw=linewidth)
    ax4.fill_between(eg_plot['temp_eg'].index,
                 eg_plot['temp_eg']-eg_plot.loc[:, "temp_eg_val"],
                 eg_plot['temp_eg']+eg_plot.loc[:, "temp_eg_val"], color='red', alpha=0.3)
    ax4.yaxis.label.set_color('red')
    ax4.tick_params(axis='y', colors='red')

    ax1.set_title(f"Flood on {pd.to_datetime(peak_date).strftime('%d %B %Y')} {str(title_suffix)}",
                  fontweight='bold', loc='left')

    plt.show()

def plot_arrow(a1, p1, a2, p2, coordsA='axes fraction', coordsB='axes fraction'):
    con = mpatches.ConnectionPatch(xyA=p1, xyB=p2, coordsA=coordsA, coordsB=coordsB,
                                   axesA=a1, axesB=a2, arrowstyle="-|>", facecolor='black')
    a1.add_artist(con)

def plot_simple_arrow(a1, p1, a2, p2, coordsA='axes fraction', coordsB='axes fraction'):
    con = mpatches.ConnectionPatch(xyA=p1, xyB=p2, coordsA=coordsA, coordsB=coordsB,
                                   axesA=a1, axesB=a2, arrowstyle="->", facecolor='black')
    a1.add_artist(con)

def plot_line(a1, p1, a2, p2, coordsA='axes fraction', coordsB='axes fraction'):
    con = mpatches.ConnectionPatch(xyA=p1, xyB=p2, coordsA=coordsA, coordsB=coordsB,
                                   axesA=a1, axesB=a2)
    a1.add_artist(con)

def plot_decomp(dataset, decomp_dict, peak_date, title_suffix=None, linewidth=1.0, figsize=(10, 5)):

    blue_colors   = mpl.cm.Blues(np.linspace(0,1,16))
    green_colors  = mpl.cm.Greens(np.linspace(0,1,16))
    red_colors    = mpl.cm.Reds(np.linspace(0,1,16))
    purple_colors = mpl.cm.Purples(np.linspace(0,1,16))
    winter_colors = mpl.cm.winter(np.linspace(0,1,16))
    autumn_colors = mpl.cm.autumn(np.linspace(0,1,16))

    decomp_plot = dataset.loc[pd.date_range(end=peak_date, periods=list(decomp_dict.values())[0]['x'].shape[0]+1, freq='d')]

    fig = plt.figure(constrained_layout=False, figsize=figsize)

    gs1 = fig.add_gridspec(nrows=2, ncols=1, hspace=1.2, left=0.000, right=0.180, top=0.70, bottom=0.30)
    gs2 = fig.add_gridspec(nrows=6, ncols=1, hspace=0.6, left=0.250, right=0.550)
    gs3 = fig.add_gridspec(nrows=3, ncols=1, hspace=0.6, left=0.650, right=1.000, top=0.80, bottom=0.20)

    ax1_1 = fig.add_subplot(gs1[0, 0])
    ax1_2 = fig.add_subplot(gs1[1, 0])

    ax2_1 = fig.add_subplot(gs2[0, 0])
    ax2_2 = fig.add_subplot(gs2[1, 0])
    ax2_3 = fig.add_subplot(gs2[2, 0])
    ax2_4 = fig.add_subplot(gs2[3, 0])
    ax2_5 = fig.add_subplot(gs2[4, 0])
    ax2_6 = fig.add_subplot(gs2[5, 0])

    ax3_1 = fig.add_subplot(gs3[0, 0])
    ax3_2 = fig.add_subplot(gs3[1, 0])
    ax3_3 = fig.add_subplot(gs3[2, 0])

    ax1_1.plot(decomp_plot['prcp'].iloc[:-1], color='k', lw=linewidth)
    ax1_2.plot(decomp_plot['tmean'].iloc[:-1], color='k', lw=linewidth)


    for i in range(16):
        ax2_1.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(peak_date)]['hi_arr'][:, i],
                   c=green_colors[i], alpha=0.60, lw=linewidth)
        ax2_2.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(peak_date)]['hc_arr'][:, i],
                   c=blue_colors[i], alpha=0.60, lw=linewidth)
        ax2_3.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(peak_date)]['hf_arr'][:, i],
                   c=red_colors[i], alpha=0.60, lw=linewidth)
        ax2_4.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(peak_date)]['ho_arr'][:, i],
                   c=purple_colors[i], alpha=0.60, lw=linewidth)
        ax2_5.plot(decomp_plot.index[:], decomp_dict[pd.to_datetime(peak_date)]['c_states'][:, i],
                   c=autumn_colors[i], alpha=0.60, lw=linewidth)
        ax2_6.plot(decomp_plot.index[:], decomp_dict[pd.to_datetime(peak_date)]['h_states'][:, i],
                   c=winter_colors[i], alpha=0.60, lw=linewidth)

        ax3_1.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(peak_date)]['h_update'][:, i],
                   c='#000', alpha=0.60, lw=linewidth*0.6)
        ax3_2.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(peak_date)]['h_forget'][:, i],
                   c='#000', alpha=0.60, lw=linewidth*0.6)

    ax3_3.bar(decomp_plot.index[:-1],
              np.matmul(decomp_dict[pd.to_datetime(peak_date)]['h_forget'][:] * decomp_dict[pd.to_datetime(peak_date)]['h_update'][:],
                        decomp_dict[pd.to_datetime(peak_date)]['dense_W'])[:, 0],
              edgecolor='k',
              width=np.timedelta64(1, 'D'),
              color='red',
              linewidth=0.6)

    ax1_1.set_xticklabels([])
    ax2_1.set_xticklabels([])
    ax2_2.set_xticklabels([])
    ax2_3.set_xticklabels([])
    ax2_4.set_xticklabels([])
    ax2_5.set_xticklabels([])
    ax3_1.set_xticklabels([])
    ax3_2.set_xticklabels([])

    ax1_1.set_title('Precipitation [mm]', loc='left', pad=0)
    ax1_2.set_title('Temperature [\u2103]',loc='left', pad=0)
    ax2_1.set_title(r'Input gate $i_t$', loc='left', pad=0)
    ax2_2.set_title(r'Candidate vector $\tilde{c}_t$', loc='left', pad=0)
    ax2_3.set_title(r'Forget gate $f_t$', loc='left', pad=0)
    ax2_4.set_title(r'Output gate $o_t$', loc='left', pad=0)
    ax2_5.set_title(r'Cell state $c_t$', loc='left', pad=0)
    ax2_6.set_title(r'Hidden state $h_t$', loc='left', pad=0)
    ax3_1.set_title(r'Information initially gained', loc='left', pad=0)
    ax3_2.set_title(r'Proportion to be retained', loc='left', pad=0)
    ax3_3.set_title(r'Information actually contributed', loc='left', pad=0)

    ax1_2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1_2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for tick in ax1_2.xaxis.get_ticklabels()[1:3] + ax1_2.xaxis.get_ticklabels()[4:6]:
        tick.set_visible(False)

    ax2_6.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2_6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for tick in ax2_6.xaxis.get_ticklabels()[1::2]:
        tick.set_visible(False)

    ax3_3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax3_3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for tick in ax3_3.xaxis.get_ticklabels()[1::2]:
        tick.set_visible(False)

    ax3_2.set_ylim(bottom=-0.1*np.percentile(decomp_dict[pd.to_datetime(peak_date)]['h_forget'][-1, :], q=0.75),
                   top=np.percentile(decomp_dict[pd.to_datetime(peak_date)]['h_forget'][-1, :], q=0.75))


    plot_simple_arrow(ax2_3, (1.01, 0.5), ax3_1, (-0.15, 0.60))
    plot_simple_arrow(ax2_4, (1.01, 0.5), ax3_1, (-0.15, 0.50))
    plot_simple_arrow(ax2_6, (1.01, 0.5), ax3_1, (-0.15, 0.40))

    plot_simple_arrow(ax2_3, (1.01, 0.5), ax3_2, (-0.10, 0.525))
    plot_simple_arrow(ax2_4, (1.01, 0.5), ax3_2, (-0.10, 0.475))

    plot_line(ax3_1, (1.02, 0.5), ax3_1, (1.08, 0.5))
    plot_line(ax3_2, (1.02, 0.5), ax3_2, (1.08, 0.5))
    plot_line(ax3_3, (1.02, 0.5), ax3_3, (1.08, 0.5))
    plot_line(ax3_1, (1.08, 0.5), ax3_3, (1.08, 0.5))
    plot_arrow(ax3_3, (1.08, 0.5), ax3_3, (1.005, 0.5))
    ax3_2.annotate(r'$\bigodot$', (1.05, -0.4), xycoords='axes fraction', backgroundcolor='white')

    fig.suptitle(f"Flood on {pd.to_datetime(peak_date).strftime('%d %B %Y')} {str(title_suffix)}",
                 fontweight='bold', x=0, ha='left')

    plt.show()
