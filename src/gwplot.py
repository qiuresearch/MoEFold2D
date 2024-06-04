import itertools
import logging
import os
import sys
import inspect
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
mpl.rcParams['agg.path.chunksize'] = 1000000000

import plotly.express as plx
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.kaleido.scope.mathjax = None # solved the problem of slow write_image()!!!
# pio.renderers.default.timeout = 3000  # Set a higher timeout (in seconds)
# os.environ['PLOTLY_RENDERER'] = 'png'
# os.environ['PLOTLY_RENDERER_TIMEOUT'] = '600'  # Set a higher timeout (in seconds)

# homebrew
import gwio
import misc

logger = logging.getLogger(__name__)

def ion() : plt.ion()
def ioff() : plt.ioff()
def show() : plt.show()


def has_display():
    """ none of these work well... """
    if os.name in ['posix'] and 'DISPLAY' not in os.environ:
        display = os.environ()

    exit_status = os.system('eog&')
    # exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    return exit_status == 0


def savefig(*filename, png=True, svg=False, eps=False, show=True, display=True, **kwargs):
    assert bool(png) + bool(svg) + bool(eps) > 0, \
        'At least one of png, svg, and eps must be provided!'

    # default_opts = {'dpi':600}

    if not filename:
        from traceback import extract_stack
        callerinfo = extract_stack()[-2]
        if not sys.stdin.isatty():
            # from IPython.core.display import Javascript
            # from IPython.display import display
            # display(Javascript('Jupyter.notebook.kernel.execute(\
            #                    "filename = " + "\'"\
            #                    +Jupyter.notebook.notebook_name+"\'");'))
            # %%javascript
            # IPython.notebook.kernel.execute('filename = "' + IPython.notebook.notebook_name + '"')
            filename = 'autoname'
        else:
            filename = os.path.splitext(os.path.basename(callerinfo[0]))[0]
    else :
        filename = filename[0]

    # filename = os.path.splitext(filename)[0]
    logger.info(f'Saving figure in PNG/SVG/EPS formats: {filename}')
    if png:
        plt.savefig(f'{filename}.png', format='png', dpi=300)
        if display or show:
            display_img(f'{filename}.png')
    if svg: plt.savefig(f'{filename}.svg', format='svg')
    if eps: plt.savefig(f'{filename}.eps', format='eps')


def display_img(img_file):
    """ call system command to display  """
    if sys.platform in ['linux', 'linux2']:
        os.system(f'eog {img_file} &')
    elif sys.platform == 'darwin':
        os.system(f'open {img_file} &')
    elif sys.platform == 'win32':
        os.system(f'open {img_file} &')


def fig_style(style='medium') :
    # plt.style.use(['seaborn-poster'])

    style = style.lower()
    if style in ['medium', 'normal', 'standard'] :
        plt.rcParams.update({'font.size': 15,
                    'figure.subplot.bottom': 0.15,
                    'figure.subplot.left' : 0.15,
                    'figure.titlesize': 23,
                    'legend.fontsize': 19,
                    'axes.labelsize': 19,
                    'axes.titlesize': 21,
                    'xtick.labelsize': 19,
                    'ytick.labelsize': 19,
                    'lines.linewidth': 3,
                    'lines.markersize': 13})
    elif style in ['tiny', 'tinyprint'] :
        plt.rcParams.update({'font.size': 9,
                    'figure.subplot.bottom': 0.12,
                    'figure.subplot.left' : 0.12,
                    'figure.titlesize': 12,
                    'legend.fontsize': 10,
                    'axes.labelsize': 10,
                    'axes.titlesize': 10,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'lines.linewidth': 1.3,
                    'lines.markersize': 3})
    elif style in ['small', 'smallprint'] :
        plt.rcParams.update({'font.size': 12,
                    'figure.subplot.bottom': 0.13,
                    'figure.subplot.left' : 0.13,
                    'figure.titlesize': 15,
                    'legend.fontsize': 15,
                    'axes.labelsize': 15,
                    'axes.titlesize': 15,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'lines.linewidth': 2,
                    'lines.markersize': 8})
    elif style in ['large', 'talk', 'paper'] :
        plt.rcParams.update({'font.size': 18,
                    'figure.subplot.bottom': 0.16,
                    'figure.subplot.left' : 0.16,
                    'figure.titlesize': 26,
                    'legend.fontsize': 22,
                    'axes.labelsize': 22,
                    'axes.titlesize': 24,
                    'xtick.labelsize': 22,
                    'ytick.labelsize': 22,
                    'lines.linewidth': 4,
                    'lines.markersize': 16})
    elif style in ['huge', 'hugeprint'] :
        plt.rcParams.update({'font.size': 21,
                    'figure.subplot.bottom': 0.17,
                    'figure.subplot.left' : 0.17,
                    'figure.titlesize': 29,
                    'legend.fontsize': 25,
                    'axes.labelsize': 25,
                    'axes.titlesize': 23,
                    'xtick.labelsize': 23,
                    'ytick.labelsize': 23,
                    'lines.linewidth': 5,
                    'lines.markersize': 18})
    elif style in ['x'] :
        pass
    else :
        pass


def screen_size(figsize='medium'):
    ''' set the plot screen to a few preset sizes'''
    mng = plt.get_current_fig_manager()

    figsize = figsize.lower()
    if figsize in ['full', 'fullscreen', 'max', 'x'] :
        backend_name = plt.get_backend().upper()
        if backend_name == 'wxAgg'.upper():
            mng.frame.Maximize(True)
        elif backend_name == 'TkAgg'.upper():
            mng.window.state('zoomed')
        elif backend_name == 'QT4Agg'.upper():
            mng.window.showMaximized()
        else:
            gwio.showinfo(infostr="Unknown backend, do NOTHING")
    elif figsize in ['small', 'crib', 'talk', 'paper', 'min'] :
        mng.resize(600,400)
    elif figsize in ['medium', 'twin', 'twinsize', 'standard', 'normal'] :
        mng.resize(800,600)
    elif figsize in ['large', 'queen', 'queensize'] :
        mng.resize(1200,800)
    elif figsize in ['huge', 'king', 'kingsize'] :
        mng.resize(1600,1200)
    else:
        gwio.showinfo(infostr="Not Implemented")


def zoomout(ax, factor):

    xlim = ax.get_xlim()
    if 'log' == ax.get_yscale():
        np.log(xlim)
        xlim = (xlim[0] + xlim[1]) / 2 + np.array((-0.5, 0.5)) * \
            (xlim[1] - xlim[0]) * (1 + factor)
        xlim = np.exp(xlim)
    else:
        xlim = (xlim[0] + xlim[1]) / 2 + np.array((-0.5, 0.5)) * \
            (xlim[1] - xlim[0]) * (1 + factor)
    ax.set_xlim(xlim)

    ylim = ax.get_ylim()
    if 'log' == ax.get_yscale():
        np.log(ylim)
        ylim = (ylim[0] + ylim[1]) / 2 + np.array((-0.5, 0.5)) * \
            (ylim[1] - ylim[0]) * (1 + factor)
        ylim = np.exp(ylim)
    else:
        ylim = (ylim[0] + ylim[1]) / 2 + np.array((-0.5, 0.5)) * \
            (ylim[1] - ylim[0]) * (1 + factor)
    ax.set_ylim(ylim)


def plot(*y, title='', xlabel='X', ylabel='Y', name=None, **kwargs) :
    """ A wrapper for plot with addition keyword argument """
    default_opts = {}
    default_opts.update(kwargs)

    plt_out = plt.plot(*y, **default_opts)
    if name is not None :
        if isinstance(name, str): name = [name]
        plt.legend(plt_out, name, frameon=False, loc='best')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return plt_out


def xy_plot(data, fmt='-', label='', **kwargs) :
    return plot(data[0], data[1], name=label, **kwargs)


def pie_plot(x, legend=True, title='', **kwargs) :
    """ legend is not shown or shown in a separate axis! """
    default_opts = {
        'startangle':0,
        'shadow':False,
        # 'explode': False,
        'autopct':'%1.1f%%',
        'labels':None,
        'pctdistance': 0.84,
        'labeldistance': 1.05,
        }

    if legend: # show a separate legend axes
        fig, ax = plt.subplots(1, 2, frameon=True, gridspec_kw={'width_ratios': [1.2, 1]})
    else :
        fig, ax = plt.subplots(frameon=True)
        ax = [ax]

    default_opts.update(kwargs)
    if legend : default_opts['labels'] = None

    wedges, texts, autotexts = ax[0].pie(x, **default_opts)
    ax[0].axis('equal')

    if legend and 'labels' in kwargs :
        ax[-1].axis('off')
        ax[-1].legend(wedges, kwargs['labels'], loc='center left', frameon=False)
        ax[0].set(xlim=(-1,1), ylim=(-1,1))
        fig.suptitle(title)
    else:
        ax[0].set_title(title)
        ax[0].set(xlim=(-1.1,1.1), ylim=(-1.1,1.1))

    return wedges, texts, autotexts


def hist_plot(x, bins=20, binwidth=None, title='', xlabel='Bins', ylabel='Counts',
              show_counts=True, show_percentage=False, fontsize=None, **kwargs) :
    """ a simple histogram plot, binwidth overides bins keywords"""
    default_opts = {'rwidth': 0.9}
    if bins is None: bins = 20
    if binwidth is not None and binwidth != 0 : # and ('bins' not in kwargs) and ('range' not in kwargs) :
        bin_grids = np.arange(np.floor(x.min()/binwidth), np.ceil(x.max()/binwidth)+1, step=1)*binwidth
    elif isinstance(bins, int): # try to create meaningful bins
        xmax = x.max()
        xmin = x.min()
        binwidth = misc.round_sig((xmax-xmin)/(bins-1), sig=1)
        bin_grids = np.arange(np.floor(xmin/binwidth), np.ceil(xmax/binwidth)+1, step=1)*binwidth
    else:
        bin_grids = bins

    kwargs.update({'bins':bin_grids})
    default_opts.update(kwargs)
    if fontsize is None:
        fontsize = 17 - len(default_opts['bins'])
    logger.info(f'Setting default fontsize: {fontsize}')
    plt.rcParams.update({
        'font.size': fontsize,
        'figure.titlesize': fontsize,
        'legend.fontsize': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'ytick.labelsize': fontsize,
        'xtick.labelsize': fontsize,
        })

    hist_data = plt.hist(x, **default_opts)

    print(default_opts)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(hist_data[1], [f'{_s:0.2f}' for _s in hist_data[1]], rotation=45)
    plt.title(f'{title}\n total:{len(x)}, min: {x.min():0.2g}, max:{x.max():.3}, mean:{x.mean():0.2g}, std:{x.std():0.2g}, median:{np.median(x):0.2g}', fontsize=fontsize)

    x_grids = (hist_data[1][:-1] + hist_data[1][1:])/2

    if show_counts : # show yvalues
        xlen = len(x)/100.0
        for i, y in enumerate(hist_data[0]) :
            plt.text(x_grids[i], y, f'{int(y)}\n({int(y/xlen)}%)',
                horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim([ylim[0], ylim[1]*1.08])
        # plt.gca().autoscale_view()

    if show_percentage:
        x_nums = np.cumsum(hist_data[0])
        # multiply max of hist_data[0] to match
        x_nums = x_nums/x_nums[-1]*np.max(hist_data[0])*1.03
        plt.step(x_grids, x_nums, fontsize=fontsize)
        # xlim = plt.gca().get_xlim()
        # plt.hlines(x_nums[-1]*np.array([0.25,0.5,0.75]), xlim[0], xlim[1], 'k', '--')

        # show twinx with different y labels
        # xlim = plt.gca().get_xlim()
        # ylim = plt.gca().get_ylim()

        # axes_twinx = plt.gca().twinx()
        # axes_twinx.set_xlim(xlim)

        # axes_twinx.set_ylabel('Percentages')


    return hist_data


def surf_plot2d(xy):
    pass


def xyerror(data, fmt='-', label=''):
    return plt.errorbar(data[:, 0], data[:, 1], data[:, 2], fmt=fmt, label=label)


def xylabel(xytype='IQ'):
    ''' set xlabel and y label'''
    # print('XYTYPE: '+xytype)
    if xytype.upper() == "IQ":
        plt.xlabel(r'$Q (\AA^{-1})$')
        plt.ylabel(r'$I(Q)$')
    elif xytype.upper() == "GUINIER":
        plt.xlabel(r'$Q^2 (\AA^{-2})$')
        plt.ylabel(r'$log(I(Q))$')
    elif xytype.upper() == "KRATKY":
        plt.xlabel(r'$Q (\AA^{-1})$')
        plt.ylabel(r'$Q^2\timesI(Q))$')
    elif xytype.upper() == "PR":
        plt.xlabel(r'$R (\AA)$')
        plt.ylabel(r'$P(R)$')
    elif xytype.upper() == "UVSPEC":
        plt.xlabel(r'$Wavelength (nm)$')
        plt.ylabel(r'$OD$')
    elif xytype.upper() == "OSMOFORCE":
        plt.xlabel(r'inter-axial spacing')
        plt.ylabel(r'log$_{10}(\Pi)$ (Pa)')
    else:
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')


def ply_save_image(fig,
        save_prefix=None,
        save_dir=None,
        auto_save_prefix=None,
        fmt=['png', 'svg'],          # format of the img (png, svg, pdf, eps)
        scale=1,                     # scale of the save img
        width=1200,
        height=800,
        trim=True,                   # trim off surrounding white space
        show=True,
        **kwargs):
    """ a generic wrapper for saving a plotly figure as an image """
    #inspect.currentframe().f_code.co_name])
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, inspect.stack()[-1].frame.f_code.co_name])
    save_prefix = os.path.join(Path(save_prefix).parent, f'{kwargs.get("save_header", "")}{Path(save_prefix).name}{kwargs.get("save_footer", "")}')
    save_path = Path(save_prefix) if save_dir is None else Path(save_dir) / save_prefix

    fmt = kwargs.get('img_fmt', kwargs.get('save_fmt', fmt))
    height = kwargs.get('img_height', kwargs.get('save_height', height))
    width = kwargs.get('img_width', kwargs.get('save_width', width))
    scale = kwargs.get('img_scale', kwargs.get('save_scale', scale))

    if isinstance(fmt, str):
        fmt = [fmt]
    for afmt in fmt:
        img_file = save_path.as_posix() + f'.{afmt}'
        logger.info(f'Saving image to: {misc.str_color(img_file)} ...')
        fig.write_image(img_file, format=afmt, width=width, height=height, scale=scale)# bbox_inches='tight')

        if trim and afmt in ['png']: # fail to create thumbnail for svg
            logger.info(f'Trimming image: {misc.str_color(img_file)} ...')
            os.system(f'convert {img_file} -trim +repage {img_file}')

        if kwargs.get('show_img', show):
            if 'png' in fmt and afmt != 'png': continue
            if img_file.endswith('.eps'):
                os.system(f'evince {img_file} &')
            else:
                os.system(f'eog {img_file} &')
            kwargs['show_img'] = False


def ply_font_families():
    """ return a list of font families supported by plotly """
    font_families = ['Arial', 'Balto', 'Courier New', 'Droid Sans', 'Droid Serif', 'Droid Sans Mono',
                     'Gravitas One', 'Old Standard TT', 'Open Sans', 'Overpass',
                     'PT Sans Narrow', 'Raleway', 'Times New Roman']
    return font_families


def ply_colors_cycle(colorway='Plotly', reverse=False):
    """ return a cycle of colors from plotly """
    colors = getattr(plx.colors.qualitative, colorway.capitalize())
    if reverse:
        return itertools.cycle(reversed(colors))
    else:
        return itertools.cycle(colors)


def ply_symbols_cycle(reverse=False):
    """ return a cycle of symbols from plotly """
    symbols = ['circle', 'square','diamond', 'triangle-up', 'pentagon', 'hexagram', 'star', 'hourglass', 'bowtie']
    if reverse:
        return cycle(reversed(symbols))
    else:
        return cycle(symbols)


def ply_dfs_xy(dfs, split_dfs=None, x=None, y=None, split_x=None, split_y=None, 
        fmt='line', names=None, texts=None, labels=None,
        groupby=None, color=None, line_group=None, nbins=None,
        title=None, showlegend=True, legend=None, panel_id=None,
        xrange=None, yrange=None, row=None, col=None, secondary_y=None,
        facet_row=False, facet_col=False, max_points=1e5,
        font_size=18, font_family="verdana, 'Open Sans', arial, sans-serif", # "Courier New, monospace",
        colors_cycle=ply_colors_cycle(), reset_colors_cycle=False,
        symbols_cycle=ply_symbols_cycle(), reset_symbols_cycle=False,
        fig=None, show=True, renderer='browser', **kwargs):
    """ list of dataframes or dicts, one x and/or one y (both col names)
        split_dfs is only allowed for violin plots

        Default behaviors for passing xtitle/ytitle:
            Not passed: using x as xtitle and ys as ytitle
            None: unchanged
            False: no xtitle or ytitle (existing ones will be cleared!)
    """
    pio.renderers.default = renderer
    if reset_colors_cycle:
        colors_cycle = ply_colors_cycle()
    if reset_symbols_cycle:
        symbols_cycle = ply_symbols_cycle()

    # ensure both dfs and names are list/tuple and of the same length
    if type(dfs) not in (list, tuple):
        dfs = [dfs]
    if split_dfs is not None:
        if type(split_dfs) not in (tuple, list):
            split_dfs = [split_dfs]
        split_plot = len(split_dfs) > 0 and any([_df is not None for _df in split_dfs])
        if split_x is None:
            split_x = x
        if split_y is None:
            split_y = y
    else:
        split_plot = False

    if names is None: # used as legends by plotly
        names = [f'{fmt}_{_i}' for _i in range(len(dfs))]
        # names = cycle([names]) if isinstance(names, str) else cycle(names)
    if type(names) not in (list, tuple):
        names = [names]
    if len(names) != len(dfs):
        logger.warning(f'len(names): {len(names)} != len(dfs): {len(dfs)}!!!')

    if texts is None: # used for texts shown on histogram and bar plots
        texts = [None] * len(dfs)
    elif type(texts) not in (list, tuple):
        texts = [texts] * len(dfs)
    if len(texts) != len(dfs):
        logger.warning(f'len(texts): {len(texts)} != len(dfs): {len(dfs)}!!!')        

    # to solve the problem of not taking row or col kwargs if not created with make_subplots()
    subplot_kwargs = {}
    if row is not None: subplot_kwargs['row'] = row
    if col is not None: subplot_kwargs['col'] = col
    # if secondary_y: subplot_kwargs['secondary_y'] = True
    # if kwargs.get('secondary_y', None) is not None: subplot_kwargs['secondary_y'] = kwargs['secondary_y']
    logger.info(f'Subplot kwargs: {subplot_kwargs}')

    if 'kwargs' in kwargs: kwargs.pop('kwargs')
    logger.info(f'kwargs: {kwargs}')
    secondary_kwargs = {}
    if secondary_y is not None:
        secondary_kwargs['secondary_y'] = secondary_y
    logger.info(f'Secondary_y kwargs: {secondary_kwargs}')

    args = misc.Struct(locals())
    logger.debug('Arguments:\n' + gwio.json_str(args.__dict__))

    if fig is None: 
        if facet_row:
            fig = make_subplots(rows=len(dfs), cols=1, vertical_spacing=0.0, subplot_titles=['TBD']*4)
        elif facet_col:
            fig = make_subplots(rows=1, cols=len(dfs), horizontal_spacing=0.0, subplot_titles=['TBD']*4)
        else:
            fig = go.Figure()

    fmt = fmt.lower()
    if showlegend is None:
        showlegend = kwargs.get('show_legend', None)
        
    logger.info(f'Adding traces with fmt: {fmt}, names: {names}...')
    for i, df in enumerate(dfs):
        if df is None:
            continue
        if max_points is not None:
            max_points = int(max_points)
            if max_points < len(df):
                logger.warning(f'Sampling {max_points} out of total {len(df)} rows for dataframe #{i+1}...')
                df = df.sample(max_points)
            if split_dfs is not None and max_points < len(split_dfs[i]):
                logger.warning(f'Sampling {max_points} out of total {len(split_dfs[i])} rows for split dataframe #{i+1}...')
                split_dfs[i] = split_dfs[i].sample(max_points)

        if facet_row:
            subplot_kwargs['row'] = i + 1
            subplot_kwargs['col'] = 1
        elif facet_col:
            subplot_kwargs['row'] = 1
            subplot_kwargs['col'] = i + 1
            
        # if y is None:
        #     if fmt not in ['hist', 'histogram', 'violin', 'pie']: # doesn't require y
        #         logger.warning('Skipping y=None column!')
        #         continue
        if fmt in ['bar']:
            fig.add_trace(go.Bar(
                x=None if x is None else df[x],
                y=None if y is None else df[y],
                name=names[i],
                error_x=dict(array=df[kwargs['error_x']]) if kwargs.get('error_x') else None,
                error_y=dict(array=df[kwargs['error_y']]) if kwargs.get('error_y') else None,
                # barmode=kwargs.get('barmode', 'group'),
                marker=dict(color=kwargs.get('marker_color', next(colors_cycle))),
                text=texts[i],
                textposition=kwargs.get('textposition', 'auto'),
                texttemplate=kwargs.get('texttemplate', '%{value}'),
                insidetextanchor=kwargs.get('insidetextanchor', 'start'),
                showlegend=showlegend,
                ),
                **subplot_kwargs,       
                )
        elif fmt in ['bar', 'hist', 'histogram', 'histxy']:
            # check if error bars are needed
            if fmt != 'bar' and (kwargs.get('error_x', None) is not None or kwargs.get('error_y', None) is not None):
                logger.warning('Error bars are not supported for histogram, using np.histogram and go.Bar!!!')
                _bar_plot = True
                if y is None:
                    y = x
                if pd.api.types.is_numeric_dtype(df[x]):
                    xbins = pd.cut(df[x], bins=kwargs.get('nbins', 11))
                    df['xbin_center'] = df.groupby(xbins)[x].transform('mean')
                    df_grps = df.groupby('xbin_center')
                    _df = df_grps[y].agg(['mean', 'std', 'count']).reset_index()
                    x = 'xbin_center'
                    y = 'mean'
                else: # x is categorical or string
                    df_grps = df.groupby(x)
                    if y == x:
                        logger.critical(f'Cannot use y={y} as x={x} for histogram with error bar!!!')
                        _df = df_grps[y].agg(['count']).reset_index()
                    else:
                        _df = df_grps[y].agg(['mean', 'std', 'count']).reset_index()
                    y = 'mean'
            else:
                _df = df
                _bar_plot = False            
                
            if fmt == 'bar' or _bar_plot:
                fig.add_trace(go.Bar(
                    x=None if x is None else _df[x],
                    y=None if y is None else _df[y],
                    name=names[i],
                    error_x=dict(array=_df[kwargs['error_x']]) if kwargs.get('error_x') else None,
                    error_y=dict(array=_df[kwargs['error_y']]) if kwargs.get('error_y') else None,
                    # barmode=kwargs.get('barmode', 'group'),
                    # marker=dict(color=kwargs.get('marker_color', next(colors_cycle))),
                    text=texts[i],
                    textposition=kwargs.get('textposition', 'auto'),
                    texttemplate=kwargs.get('texttemplate', '%{value:.3g}'),
                    insidetextanchor=kwargs.get('insidetextanchor', 'start'),
                    showlegend=showlegend,
                    # labels=labels, # not supported
                    ),
                    **subplot_kwargs,       
                    )
            else:
                fig.add_trace(go.Histogram(
                    x=None if x is None else df[x],
                    y=None if y is None else df[y],
                    histnorm=kwargs.get('histnorm', None), #'percent'),
                    histfunc=kwargs.get('histfunc', 'avg'),
                    name=names[i],
                    # marker=dict(color=kwargs.get('marker_color', next(colors_cycle))),
                    textposition=kwargs.get('textposition', 'auto'),
                    texttemplate=kwargs.get('texttemplate', '%{value:.3g}'),
                    cumulative_enabled=False,
                    # xbins=dict(
                    #     start=-10.0,
                    #     end=10.0,
                    #     size=1,
                    # ),
                    # marker_color='blue',
                    # opacity=1.0,
                    ),
                    **subplot_kwargs,
                    )                
        elif fmt == 'line':
            fig.add_trace(go.Scatter(
                x=None if x is None else df[x],
                y=None if y is None else df[y],
                mode=kwargs.get('mode', 'lines+markers'),
                line=dict(width=kwargs.get('line_width', 4),
                          dash=kwargs.get('line_dash', None), 
                          color=kwargs.get('line_color', next(colors_cycle))
                          ),
                marker=dict(symbol=next(symbols_cycle), size=kwargs.get('marker_size', 11)),
                fillcolor=kwargs.get('fillcolor', None),
                line_shape='linear',  # 'spline', 'vhv', 'hvh', 'hv', 'vh'
                name=names[i],
                showlegend=showlegend,
                ),
                **secondary_kwargs,
                **subplot_kwargs,
                )
        elif fmt == 'scatter':
            if len(df) > 2e5: # 2e4:
                fig.add_trace(go.Scattergl(
                    x=df[x], y=df[y],
                    mode=kwargs.get('mode', 'markers'),
                    text=kwargs.get('text', None),
                    marker=dict(
                        symbol=kwargs.get('marker_symbol', next(symbols_cycle)), 
                        size=kwargs.get('marker_size', 11), 
                        color=kwargs.get('marker_color', next(colors_cycle))),
                    fillcolor=kwargs.get('fillcolor', None),
                    name=names[i],
                    showlegend=showlegend,
                    ),
                    **secondary_kwargs,
                    **subplot_kwargs,
                    )
            else:
                fig.add_trace(go.Scatter(
                    x=df[x], y=df[y],
                    mode=kwargs.get('mode', 'markers'),
                    text=kwargs.get('text', None),
                    marker=dict(
                        symbol=kwargs.get('marker_symbol', next(symbols_cycle)), 
                        size=kwargs.get('marker_size', 11), 
                        color=kwargs.get('marker_color', next(colors_cycle))),
                    fillcolor=kwargs.get('fillcolor', None),
                    name=names[i],
                    showlegend=showlegend,
                    ),
                    **secondary_kwargs,
                    **subplot_kwargs,
                    )

        elif fmt == 'box':
            fig.add_trace(go.Box(
                x=None if x is None else df[x],
                y=None if y is None else df[y],                
                boxpoints=kwargs.get('boxpoints', 'all'),
                jitter=kwargs.get('jitter', 0.2),
                pointpos=kwargs.get('pointpos', 0.0),
                showlegend=showlegend,
            ))
        elif fmt == 'violin':
            fig.add_trace(go.Violin(
                x=None if x is None else df[x],
                y=None if y is None else df[y],
                spanmode=kwargs.get('spanmode', 'manual'),
                span=kwargs.get('span', [0,1]),
                width=kwargs.get('width', 0), # in data coordinates (0 means auto)
                name=names[i],
                box_visible=kwargs.get('box_visible', False),
                box_width=kwargs.get('box_width', 0.2), # relative to the violin's width
                # box_fillcolor='white',
                meanline_visible=kwargs.get('meanline_visible', True),
                meanline_color=kwargs.get('meanline_color', "white"),
                meanline_width=kwargs.get('meanline_width', 3),
                points=kwargs.get('points', 'suspectedoutliers'),
                side='negative' if split_plot else None,
                fillcolor=kwargs.get('fillcolor', kwargs.get('line_color', 'tan')) if split_plot else None, # 'chocolate',
                line_color=kwargs.get('line_color', kwargs.get('fillcolor', 'tan')) if split_plot else None,
                opacity=kwargs.get('opacity', 1.0),
                scalegroup='L' if kwargs.get('scalegroup', False) else f'L{i}', # will default to data with the same name
                showlegend=False,
                ),
                **subplot_kwargs,
                )
            if split_plot and split_dfs[i] is not None:
                fig.add_trace(go.Violin(
                    x=None if split_x is None else split_dfs[i][split_x],
                    y=None if split_y is None else split_dfs[i][split_y],
                    spanmode=kwargs.get('spanmode', 'manual'),
                    span=kwargs.get('span', [0,1]),
                    width=kwargs.get('width', 0), # in data coordinates
                    name=names[i],
                    box_visible=kwargs.get('box_visible', False),
                    box_width=kwargs.get('box_width', 0.2), # relative to the violin's width
                    # box_fillcolor='white',
                    meanline_visible=kwargs.get('meanline_visible', True),
                    meanline_color=kwargs.get('meanline_color', "white"),
                    meanline_width=kwargs.get('meanline_width', 3),
                    points=kwargs.get('points', 'suspectedoutliers'),
                    side="positive",
                    fillcolor=kwargs.get('split_fillcolor', kwargs.get('split_line_color', 'blue')),
                    line_color=kwargs.get('split_line_color', kwargs.get('split_fillcolor', 'blue')),
                    opacity=kwargs.get('split_opacity', 0.6),
                    scalegroup='R' if kwargs.get('scalegroup', False) else f'R{i}',
                    showlegend=False,
                    ),
                    **subplot_kwargs,
                    )

        elif fmt == 'sunburst':
            fig.add_trace(go.Sunburst(
                ids=df['ids'] if 'ids' in df else None,
                labels=df['labels'] if 'labels' in df else None,
                parents=df['parents'] if 'parents' in df else None,
                values=df['values'] if 'values' in df else None,
                branchvalues=kwargs.get('branchvalues', 'total'),
                ),
                **subplot_kwargs,
                )
        else:
            logger.error(f'Unknown fmt: {fmt}!!!')

    logger.info(f'Updating layouts...')

    # global settings
    if legend is None:
        legend = dict(
            title_font_family=kwargs.get('legend_title_font_family', font_family),
            font=dict(
                size=kwargs.get('legend_font_size', font_size),
            ), 
        )
    fig.update_layout(
        title=title,
        showlegend=showlegend,
        legend=legend,
        # legend_title="Mol. Type",
        # xaxis_title=x,
        # xaxis=dict(
        #     title=None if xtitle is False else xtitle if xtitle is not None else \
        #         fig.layout.xaxis.title if fig.layout.xaxis.title else x,
            # type=kwargs.get('xlog', None),
            # linecolor="#BCCCDC",
            # showgrid=False,
            # showspikes=True, # Show spike line for X-axis
            # # Format spike
            # spikethickness=2,
            # spikedash="dot",
            # spikecolor="#999999",
            # spikemode="across",
        # ),
        # yaxis=dict(
            # title=kwargs.get('ytitle', y),
            # range=yrange,
            # linecolor="#BCCCDC",
            # showgrid=False,
        # ),
        font=dict(
            family=font_family,
            size=font_size,
            color='Black', # RebeccaPurple
        ),
    )

    if title is not None:
        max_chars_per_line = max([len(line) for line in title.lower().split('<br>')]) + 1
        title_font_size = 200 * kwargs.get('img_width', 1200) / 1200 // np.sqrt(max_chars_per_line)
        fig.layout.title.font.size = kwargs.get('title_font_size', title_font_size)

    xaxes_opts = dict(
        range=xrange,
        **subplot_kwargs,
        titlefont=dict(size=font_size),
        tickfont=dict(size=font_size),
        categoryorder='array' if kwargs.get('categoryarray', None) else None,
    )
    if kwargs.get('categoryarray') is not None:
        xaxes_opts.update(categoryarray=kwargs['categoryarray'])

    # check if the x axis is categorical
    if labels is not None and x is not None and pd.api.types.is_categorical_dtype(df[x]):
        xaxes_opts['type'] = 'category'
        xaxes_opts['categoryorder'] = 'array'
        xaxes_opts['ticktext'] = [labels.get(_s, _s) for _s in fig.layout.xaxis.ticktext]

    if kwargs.get('xtitle', x) is not None:
        xaxes_opts['title_text'] = None if kwargs.get('xtitle', None) is False else kwargs.get('xtitle', x)
    fig.update_xaxes(**xaxes_opts)
    # if xrange is not None:
        # fig.layout.xaxis.range = xrange

    yaxes_opts = dict(
        range=yrange,
        titlefont=dict(size=font_size),
        tickfont=dict(size=font_size),        
        **secondary_kwargs,
        **subplot_kwargs,
    )
    if kwargs.get('ytitle', y) is not None:
        yaxes_opts['title_text'] = None if kwargs.get('ytitle', None) is False else kwargs.get('ytitle', y)
    fig.update_yaxes(**yaxes_opts)

    # fmt-specific settings
    if fmt == 'violin':
        # update characteristics shared by all traces
        fig.update_traces(
            meanline_visible=kwargs.get('meanline_visible', True),
            scalemode=kwargs.get('scalemode', 'width'), # or "count" for scaling violin plot area with total count
            # points='all', # show all points
            # jitter=0.1,  # add some jitter on points only (not for violin) for better visibility
            selector=dict(type='violin'),
            )
        fig.update_layout(
            violingap=kwargs.get('violingap', 0.07),  # gap between violins with different x (only if width is unset)
            violingroupgap=kwargs.get('violingroupgap', 0.05), #  gap between violins/points with the same x (only if width is unset)
            violinmode=kwargs.get('violinmode', 'overlay'), # group or overlap, only for violins with the same x
            )
    elif fmt in ['bar', 'hist', 'histogram']:
        fig.update_layout(
            barmode=kwargs.get('barmode', 'group'), # stack/overlay/group
            bargap=kwargs.get('bargap', 0.15),
            bargroupgap=kwargs.get('bargroupgap', 0.1),
            xaxis=dict(
                showgrid=True,
                ticks="outside",
                tickson="boundaries",
                categoryorder=kwargs.get('categoryorder'), # choose from ['trace', 'category ascending', 'category descending', 'array', 'total ascending', 'total descending', 'min ascending', 'min descending', 'max ascending', 'max descending', 'sum ascending', 'sum descending', 'mean ascending', 'mean descending', 'median ascending', 'median descending']
            ),
        )
        if kwargs.get('category_orders') is not None or kwargs.get('categoryarray') is not None:
            fig.update_xaxes(categoryorder='array',
                             categoryarray=kwargs.get('categoryarray', 
                                                      kwargs['category_orders'][x] if isinstance(kwargs['category_orders'], dict) else kwargs['category_orders']))
        fig.update_yaxes(
            type='log' if kwargs.get('ylog', None) else None,
        )
    elif fmt in ['line', 'scatter']:
        pass
        # fig.update_traces(textinfo=kwargs.get('textinfo', 'auto'))

    # Add panel_id
    # annotations = list(fig['layout']['annotations'])
    # fig.update_layout(annotations=annotations)
    if panel_id is not None:
        fig.add_annotation(
            xref='paper', x=kwargs.get('panel_id_x', -0.05), xanchor='center',
            yref='paper', y=kwargs.get('panel_id_y', 0.92), yanchor='middle',
            text=panel_id,
            font=dict(family="verdana, 'Open Sans', arial, sans-serif",
                      size=kwargs.get('panel_id_font_size', font_size + 15),
                    #   color='rgb(150,150,150)',
                      ),
            showarrow=False)

    if show: fig.show(config={
        "displayModeBar": kwargs.get('displayModeBar', True),
        "showTips": kwargs.get('showTips', True)
        })
    return fig


def symbol_order(i, offset=False):
    symbols = ['s', 'o', '^', 'd', 'v', '<', 'p', 'h']

    if offset:
        return symbols[i % len(symbols)] + offset
    else:
        return symbols[i % len(symbols)]


def color_order(i):
    # colors = [[0,    0,    1],
    #           [0,  0.5,    0],
    #           [1,    0,    0],
    #           [0.75,    0, 0.75],
    #           [0, 0.75, 0.75],
    #           [0.75, 0.75,    0],
    #           # [   0,    1,    0],
    #           [0.25,    0, 0.25],
    #           # [   1,    1,    0],
    #           [1,  0.5,    0]]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors[i % len(colors)]


def qual_color(i, style='set4'):
    '''
    http://www.personal.psu.edu/cab38/ColorBrewer/ColorBrewer.html
    9-class qualitative Set1,
    '''
    # solarized
    solarized = [[ 38, 139, 210], # blue      #268bd2  4/4 blue      33 #0087ff
                 [220,  50,  47], # red       #dc322f  1/1 red      160 #d70000
                 [133, 153,   0], # green     #859900  2/2 green     64 #5f8700
                 [211,  54, 130], # magenta   #d33682  5/5 magenta  125 #af005f
                 [181, 137,   0], # yellow    #b58900  3/3 yellow   136 #af8700
                 [ 42, 161, 152], # cyan      #2aa198  6/6 cyan      37 #00afaf
                 [108, 113, 196], # violet    #6c71c4 13/5 brmagenta 61 #5f5faf
                 [203,  75,  22], # orange    #cb4b16  9/3 brred    166 #d75f00
                 [131, 148, 150], # base0     #839496 12/6 brblue   244 #808080
                 [  0,  43,  54]] # base03    #002b36  8/4 brblack  234 #1c1c1c

    # 12-class paired
    pair = [[166, 206, 227],
            [31, 120, 180],
            [178, 223, 138],
            [51, 160,  44],
            [251, 154, 153],
            [227,  26,  28],
            [253, 191, 111],
            [255, 127,   0],
            [202, 178, 214],
            [106,  61, 154],
            [255, 255, 153],
            [177,  89,  40]]

    # 9-class Set1
    set1 = [[228,  26,  28],
            [55,  126, 184],
            [77,  175,  74],
            [152,  78, 163],
            [255, 127,   0],
            [255, 255,  51],
            [166,  86,  40],
            [247, 129, 191],
            [153, 153, 153]]

    # 8-class Dark2
    dark = [[27, 158, 119],
            [217,  95,   2],
            [117, 112, 179],
            [231,  41, 138],
            [102, 166,  30],
            [230, 171,   2],
            [166, 118,  29],
            [102, 102, 102]]


    #  http://tools.medialab.sciences-po.fr/iwanthue/
    set2 = [[206, 158, 154],
            [127, 206, 78],
            [201, 80, 202],
            [210, 89, 53],
            [78, 96, 56],
            [77, 59, 89],
            [138, 206, 163],
            [204, 178, 76],
            [199, 76, 125],
            [130, 173, 197],
            [117, 59, 44],
            [135, 115, 198]]

    set3 = [[24, 158, 174],
            [249, 117, 53],
            [212, 13, 158],
            [180, 192, 18],
            [79, 3, 41],
            [100, 101, 23],
            [31, 45, 57],
            [234, 126, 247],
            [29, 98, 167],
            [146, 94, 238]]

    set4 = [[56, 97, 138],
            [217, 61, 62],
            [75, 108, 35],
            [100, 68, 117],
            [228, 182, 48],
            [183, 92, 56],
            [107, 171, 215],
            [209, 84, 175],
            [177, 191, 57],
            [126, 116, 209]
            ]

    set5 = [[194, 141, 57],
            [173, 95, 211],
            [78, 156, 139],
            [108, 173, 68],
            [97, 77, 121],
            [166, 82, 84],
            [84, 94, 43],
            [202, 82, 147],
            [205, 73, 52],
            [128, 147, 203]]

    mpl_set = None #plt.cm.Set3(np.linspace(0, 1, 12))[:, :3] * 255.0
    #mpl_set = np.linspace(0, 1, 12)[:, :3] * 255.0
    styles = {'set1': set1, 'pair': pair, 'dark': dark, 'set2': set2, 'solarized': solarized,
              'set3': set3, 'set4': set4, 'set5': set5, 'mpl_set': mpl_set}

    colors = styles[style]

    return np.array(colors[i % len(colors)]) / 255.0


def seq_color(i, sort=True):
    '''
    http://geog.uoregon.edu/datagraphics/color_scales.htm
    Stepped-sequential scheme, 5 hues x 5 saturation/value levels
    '''
    colors = [[0.600, 0.060, 0.060],
              [0.700, 0.175, 0.175],
              [0.800, 0.320, 0.320],
              [0.900, 0.495, 0.495],
              [1.000, 0.700, 0.700],
              [0.600, 0.330, 0.060],
              [0.700, 0.438, 0.175],
              [0.800, 0.560, 0.320],
              [0.900, 0.697, 0.495],
              [1.000, 0.850, 0.700],
              [0.420, 0.600, 0.060],
              [0.525, 0.700, 0.175],
              [0.640, 0.800, 0.320],
              [0.765, 0.900, 0.495],
              [0.900, 1.000, 0.700],
              [0.060, 0.420, 0.600],
              [0.175, 0.525, 0.700],
              [0.320, 0.640, 0.800],
              [0.495, 0.765, 0.900],
              [0.700, 0.900, 1.000],
              [0.150, 0.060, 0.600],
              [0.262, 0.175, 0.700],
              [0.400, 0.320, 0.800],
              [0.562, 0.495, 0.900],
              [0.750, 0.700, 1.000]]
    if sort:
        j, k = divmod(i, 5)
        i = 5 * k + j

    return colors[i % len(colors)]


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    Example:
        rvb = make_colormap([c('red'), c('violet'), 0.33, c('violet'),
            c('blue'), 0.66, c('blue')])
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def diverge_map(low=qual_color(0), high=qual_color(1)):
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, str):
        low = c(low)
    if isinstance(high, str):
        high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])

    # def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    # c = mcolors.ColorConverter().to_rgb
    # if isinstance(low, basestring): low = c(low)
    # if isinstance(high, basestring): high = c(high)
    # return make_colormap([low, c('white'), 0.5, c('white'), high])


def add_inset(ax, rect, axisbg='w'):
    '''
    taken from http://stackoverflow.com/a/17479417/3585557
    '''
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def example1():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    rect = [0.2, 0.2, 0.7, 0.7]
    ax1 = add_inset(ax, rect)
    ax2 = add_inset(ax1, rect)
    ax3 = add_inset(ax2, rect)
    plt.show()


def example2():
    fig = plt.figure(figsize=(10, 10))
    axes = []
    subpos = [0.2, 0.6, 0.3, 0.3]
    x = np.linspace(-np.pi, np.pi)
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i))
    for axis in axes:
        axis.set_xlim(-np.pi, np.pi)
        axis.set_ylim(-1, 3)
        axis.plot(x, np.sin(x))
        subax1 = add_inset(axis, subpos)
        subax2 = add_inset(subax1, subpos)
        subax1.plot(x, np.sin(x))
        subax2.plot(x, np.sin(x))


def example3():
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11, 4))
    gs1 = GridSpec(1, 2, left=0.075, right=0.75, wspace=0.1, hspace=0,
                   top=0.95)
    ax1 = plt.subplot(gs1[:, 0])
    ax_in1 = add_inset(ax1, [0.1, 0.2, 0.3, 0.3], 'g')
    ax_in2 = add_inset(ax1, [0.5, 0.6, 0.3, 0.3], 'b')


def main() :
    # matplotlib.use('Agg')

    title = 'Default Matplotlib Settings (from matplotlibrc file)'
    j = 11
    x = np.array(range(j))
    y = np.ones(j)
    fig = plt.figure()
    for i in x:
        y[:] = i
        s = symbol_order(i, '-')
        plt.plot(x, y, s, ms=10, label='default color: %d' %i, linewidth=5)
    dy = 0.1
    plt.ylim([-dy, x[-1] + dy])
    leg = plt.legend(scatterpoints=1, numpoints=1)
    name = 'default_colors'
    plt.title(title)
    fig.savefig('%s.eps' % name)
    fig.savefig('%s.png' % name)
    plt.show()

    title = 'Duplicating MATLAB Color Settings'
    j = 11
    x = np.array(range(j))
    y = np.ones(j)
    fig = plt.figure()
    for i in x:
        y[:] = i
        s = symbol_order(i, '-')
        plt.plot(x, y, s, mec=color_order(i), c=color_order(i), ms=10,
                 mfc='none', label=str(color_order(i)), linewidth=5)
        # plt.plot(x, y, s, ms = 10)
        # plt.scatter(x, y, s, markeredgecolor=color_order(i), facecolors='none')
    dy = 0.1
    plt.ylim([-dy, x[-1] + dy])
    leg = plt.legend(scatterpoints=1, numpoints=1)
    name = 'python_symbol_color'
    plt.title(title)
    fig.savefig('%s.eps' % name)
    fig.savefig('%s.png' % name)
    plt.show()

    title = 'Qualitative Colors'
    j = 12
    x = np.array(range(j))
    y = np.ones(j)
    fig = plt.figure()
    style = 'set1'
    style = 'dark'
    style = 'set2'
    style = 'set4'
    # style = 'mpl_set'
    # style = 'solarized'
    for i in x:
        y[:] = i
        s = symbol_order(i, '-')
        plt.plot(x, y, s, mec=qual_color(i, style), c=qual_color(i, style), ms=15,
                 mfc='none', label=str(qual_color(i, style)), linewidth=10)
        # plt.plot(x, y, s, ms = 10)
        # plt.scatter(x, y, s, markeredgecolor=color_order(i, style), facecolors='none')
    dy = 0.1
    plt.ylim([-dy, x[-1] + dy])
    leg = plt.legend(scatterpoints=1, numpoints=1)
    name = 'qual_color_' + style
    plt.title(title)
    fig.savefig('%s.eps' % name)
    fig.savefig('%s.png' % name)
    plt.show()

    title = 'Sequential Colors'
    j = 25
    x = np.array(range(j))
    y = np.ones(j)
    fig = plt.figure()
    sort = False
    for i in x:
        y[:] = i
        s = symbol_order(i, '-')
        plt.plot(x, y, s, mec=seq_color(i, sort), c=seq_color(i, sort), ms=10,
                 mfc='none', label=str(seq_color(i, sort)), linewidth=2)
        # plt.plot(x, y, s, ms = 10)
        # plt.scatter(x, y, s, markeredgecolor=color_order(i), facecolors='none')
    dy = 0.1
    plt.ylim([-dy, x[-1] + dy])
    leg = plt.legend(scatterpoints=1, numpoints=1)
    if sort:
        name = 'sorted_seq_color'
    else:
        name = 'seq_color'
    plt.title(title)
    fig.savefig('%s.eps' % name)
    fig.savefig('%s.png' % name)
    plt.show()

if __name__ == '__main__':
    main()
