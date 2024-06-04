#!/usr/bin/env python
import argparse
import os
import inspect
import itertools
import logging
import multiprocessing as mp
# mp.set_start_method('loky')
import multiprocessing.pool as mpp
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import colorlog
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# istarmap.py for Python 3.8+
# Copied from https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap (map over an interable)
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpi.pool.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap
import multiprocessing as mpi


class MainError(Exception):
    pass


class DefaultDict(defaultdict):
    """ this enables constructs like: d = DefaultDict(lambda key: key + 5) """
    def __missing__(self, key):
        return self.default_factory(key)


class Struct(object):
    def __init__(self, in_dict=dict(), **kwargs):
        self.__dict__.update(in_dict)
        self.__dict__.update(kwargs)

    def __str__(self):
        print(self.__dict__)

    def __repr__(self):
        return(str(self.__dict__))

    def __len__(self):
        return(len(self.__dict__))

    def __getitem__(self, key, default=None):
        return(self.__dict__.get(key, default))

    def __contains__(self, key):
        return(key in self.__dict__)
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def __delitem__(self, key):
        del self.__dict__[key]

    def __getattr__(self, key, default=None):
        return(self.__dict__.get(key, default))
    
    def __setattr__(self, key, val):
        self.__dict__[key] = val

    def __delattr__(self, key, default=None):
        self.__dict__.pop(key, default)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def update(self, in_dict=dict(), skip_existing=False, **kwargs):
        in_dict.update(kwargs)
        if len(in_dict) >0:
            if skip_existing:
                for key, val in in_dict.items():
                    if key not in self.__dict__:
                        self.__dict__[key] = val
                    # self.__dict__.setdefault(key, val)            
            else:
                self.__dict__.update(in_dict)

        return self


def dict_random_split(dict_in, size=0.1, return_idx=False):
    r"""Split each key values like train_test_split by treating every value as list/tuple.
    Perhaps better deal this with pandas.
    """
    num_data = len(dict_in[list(dict_in.keys())[0]])

    # size can be a number or a fraction
    if 0.0 < size < 1.0:
        size = int(num_data * size)
    else:
        size = int(size)

    # make sure size < num_data / 2
    if size > num_data / 2:
        size = num_data - size
        reverse_order = True
    else:
        reverse_order = False

    indices = np.sort(np.random.choice(num_data, size, replace=False))

    dict_out1, dict_out2 = dict(), dict()

    for key, val in dict_in.items():
        dict_out1[key] = []
        dict_out2[key] = []

        dict_out2[key].extend(val[:indices[0]])

        for i in range(size - 1):
            dict_out1[key].append(val[indices[i]])
            dict_out2[key].extend(val[indices[i] + 1:indices[i + 1]])

        dict_out1[key].append(val[indices[-1]])
        dict_out2[key].extend(val[indices[-1] + 1:])

    if reverse_order:
        return dict_out2, dict_out1
    else:
        return dict_out1, dict_out2


def dict_reverse_keys(dict_in):
    dict_out = {}
    for key, val in dict_in.items():
        dict_out[key[::-1]] = val
    return dict_out


def dict_add_reversed_keys(dict_in):
    dict_in.update(dict_reverse_keys(dict_in))
    return dict_in


def dict_add_prefix(dict_in, prefix):
    dict_out = {}
    for key, val in dict_in.items():
        dict_out[prefix + key] = val
    return dict_out


def dict_add_suffix(dict_in, suffix):
    dict_out = {}
    for key, val in dict_in.items():
        dict_out[key + suffix] = val
    return dict_out


def dict_add_prefix_suffix(dict_in, prefix, suffix):
    dict_out = {}
    for key, val in dict_in.items():
        dict_out[prefix + key + suffix] = val
    return dict_out


def dict_unzip_split(dict_in):
    """ unzip a dict of lists/tuples into a dict of lists/tuples """
    num_dicts = None
    for key, val in dict_in.items():
        if num_dicts is None:
            num_dicts = len(val)
            dicts_out = [{} for _ in range(num_dicts)]

        for i in range(num_dicts):
            dicts_out[i][key] = val[i]
    return dicts_out


def dicts_aggregate(fn, dicts, dtype=None):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_vs = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dicts_aggregate(fn, all_vs, dtype=dtype)
        elif isinstance(v, dtype):
            new_dict[k] = fn(all_vs)
        else:
            new_dict[k] = all_vs

    return new_dict


def get_1st_value(val_list, default=None, last=False):
    """ return the first no None values in the list """
    if last: val_list = list(reversed(val_list))

    for val in val_list:
        if val is not None: return val

    return default


def auto_bin_edges(xmin, xmax, nbins=10, binsize=None, return_dict=False):

    if binsize is not None and binsize != 0 : # and ('bins' not in kwargs) and ('range' not in kwargs) :
        bin_edges = np.arange(np.floor(xmin/binsize), np.ceil(xmax/binsize)+1, step=1)*binsize
    elif isinstance(nbins, int) : # try to create meaningful bins
        binsize = round_sig((xmax-xmin)/(nbins-1), sig=1)
        bin_edges = np.arange(np.floor(xmin/binsize), np.ceil(xmax/binsize)+1, step=1)*binsize

    if return_dict:
        return dict(start=bin_edges[0], end=bin_edges[-1], size=bin_edges[1] - bin_edges[0])
    else:
        return bin_edges


def mpi_on():
    return True if mpi.current_process().daemon else False


def mpi_map(hfunc, data_in, num_cpus=0.8, total=None, chunksize=None, starmap=False, 
            desc='Multiprocessing', quiet=False, tqdm_disable=True):
    """  """
    if 0 < num_cpus < 1:
        num_cpus = round(mpi.cpu_count() * num_cpus)
    elif num_cpus > mpi.cpu_count():
        num_cpus = mpi.cpu_count() - 1
    else:
        logger.error(f'Invalid num_cpus: {num_cpus}! Default to 42%...')
        num_cpus = round(mpi.cpu_count() * 0.42)

    if hasattr(data_in, '__len__') or total is not None:
        if total is None:
            total = len(data_in)
        if chunksize is None:
            chunksize, extra = divmod(total, num_cpus * 4)
            if extra: chunksize += 1
            chunksize = min([2048, chunksize])
        num_chunks = total // chunksize
    else:
        num_chunks = None
    
    if mpi_on():
        logger.debug(f'Cannot start another MPI pool within an active MPI process!!!')
        if starmap:
            data_out = itertools.starmap(hfunc, data_in)
        else:
            data_out = map(hfunc, data_in)
        
        return list(data_out)

    if not quiet:
        logger.info(f'{desc} (total={total}, nCPU={num_cpus}, nChunks~{num_chunks}, chunksize={chunksize}) ...')

    with mpi.Pool(processes=num_cpus) as mpool:
        if tqdm_disable or total is None or total < 2:
            if starmap:
                data_out = mpool.starmap(hfunc, data_in, chunksize=chunksize)
            else:
                data_out = mpool.map(hfunc, data_in, chunksize=chunksize)
        else:
            if starmap:
                data_out = mpool.istarmap(hfunc, data_in, chunksize=chunksize)
            else:
                data_out = mpool.imap(hfunc, data_in, chunksize=chunksize)

            data_out = list(tqdm(data_out, total=total, desc=desc))
        
    return data_out


def mpi_starmap(hfunc, data_in, num_cpus=0.8, desc='Multiprocessing', quiet=False):
    """  """
    if 0 < num_cpus < 1:
        num_cpus = round(mpi.cpu_count() * num_cpus)
    elif num_cpus > mpi.cpu_count():
        num_cpus = mpi.cpu_count() - 1

    if not quiet:
        logger.info(f'{desc} (nCPU={num_cpus}) ...')
    with mpi.Pool(processes=num_cpus) as mpool:
        data_out = mpool.starmap(hfunc, data_in)
    mpool.close()

    return data_out


def str_isnumeric(inp, finite=True):
    inp = inp.strip()
    try:
        val = float(inp)
        if finite:
            return np.isfinite(val)
        return True
    except:
        return False


def str_find_all(sup_str, sub_str):
    """ find the start index of all occurences of sub_str
    an empty list is returned if none found
    """
    idx = [sup_str.find(sub_str)]
    while idx[-1] != -1:
        idx.append(sup_str.find(sub_str, idx[-1] + 1))
    idx.pop(-1)
    return idx


def str_diff(str1, str2, rmv_prefix=True, rmv_suffix=True):
    """ remove common prefix and suffix from two strings """
    max_len = len(str1) if len(str1) < len(str2) else len(str2)

    if rmv_prefix:
        for i1 in range(max_len):
            if str1[i1] != str2[i1]:
                break        
            else:
                i1 += 1
    else:
        i1 = 0

    i2 = -1
    if rmv_suffix:
        for i2 in range(-1, -max_len-1+i1, -1):
            if str1[i2] != str2[i2]:
                break        

    if i2 == -1:
        return str1[i1:], str2[i1:]
    else:
        return str1[i1:i2+1], str2[i1:i2+1]


def str_deblank(str_in):
    """ see https://stackoverflow.com/questions/3739909/how-to-strip-all-whitespace-from-string """
    return re.sub(r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+', '', str_in)


def str_center(str_in, cen_char, fill_char=' ', fill_end=True):
    if isinstance(cen_char, int): # length
        return str_in.center(cen_char)
    elif isinstance(cen_char, str):
        left_len = str_in.find(cen_char)
        right_len = len(str_in) - left_len - len(cen_char)
        new_len = max([left_len, right_len])
        if fill_end:
            return fill_char*(new_len - left_len) + str_in + fill_char*(new_len - right_len)
        else: # fill both sides of the cen_char
            return str_in[:left_len] + fill_char*(new_len - left_len) + cen_char + fill_char*(new_len - right_len) + str_in[-right_len:]
    else:
        return str_in


def str_color(str_in, color='light_yellow', bkg='none', style='normal'):
    r""" escape char: \033 \e \\x1B  """
    style_dict = defaultdict(lambda: 0)
    style_dict.update(dict(
        normal = 0,
        bold = 1,
        dim = 2,
        underline = 4,
        blink = 5,
        reverse = 7,
        hidden = 8,
    ))

    fg_color_dict = defaultdict(lambda: 39)
    fg_color_dict.update(dict(
        black = 30,
        red = 31,
        green = 32,
        yellow = 33,
        blue = 34,
        magenta = 35,
        cyan = 36,
        light_gray = 37,
        dark_gray = 90,
        light_red = 91,
        light_green = 92,
        light_yellow = 93,
        light_blue = 94,
        light_magenta = 95,
        light_cyan = 96,
        white = 97))
        
    bg_color_dict = defaultdict(lambda: 49)
    for k, v in fg_color_dict.items():
        bg_color_dict[k] = v + 10

    # \033[0;46m{args.save_dir}\033[0m'
    str_out = f'\033[{style_dict[style]};{fg_color_dict[color]};{bg_color_dict[bkg]}m{str_in}\033[0m'

    return str_out


def str_auto_value(str_in):
    try:
        val = float(str_in)
        if int(val) == val:
            val = int(val)
    except:
        val = str2bool(str_in, raise_error=False)
    
    return val


def str2int(str_in, default=0):
    try:
        val = int(str_in)
    except:
        str_in = re.sub("[^0123456789-]", '', str_in) # remove all non digits/-
        try:
            val = int(str_in)
        except:
            val = default
    
    return val


def str2float(str_in, default=0.0):
    try:
        val = float(str_in)
        # if int(val) == val:
            # val = int(val)
    except:
        str_in = re.sub("[^eE0123456789.-]", '', str_in) # remove all non digits/./-/e
        try:
            val = float(str_in)
        except:
            val = default
    
    return val
    

def str2bool(v: str, raise_error=False):
    """ str is returned as-is if cannot convert to bool. Copied from stackflow """
    if isinstance(v, bool):
        return v
    vlow = v.lower()
    if vlow in ('yes', 'true', 't', 'y', '1'):
        return True
    elif vlow in ('no', 'false', 'f', 'n', '0'):
        return False
    elif vlow in ('none', 'null', 'nil', 'nill'):
        return None
    else:
        if raise_error:
            raise ValueError(f'Cannot convert {vlow} to bool!')
            # raise argparse.ArgumentTypeError('Boolean value expected')
        else:
            return v


def locate_num(val_list, val, exact=False):
    """ return the index of the item closest to val """
    if not hasattr(val_list, '__len__'):
        return 0
    if not isinstance(val_list, np.ndarray):
        val_list = np.array(val_list)
    return np.abs(val_list - val).argmin()


def logging_LogRecord(*args, **kwargs):
    record = logging.LogRecord(*args, **kwargs)
    # if record.module.__class__ == 'method':
    # for m in module: print(m.__qualname__)

    caller_frame = inspect.stack()[4].frame
    if 'self' in caller_frame.f_locals:
        record.funcName = \
            f"{caller_frame.f_locals['self'].__class__.__name__}.{record.funcName}"

    return record


class ColorFormatter(logging.Formatter):
    grey = "\\x1B[38;21m"
    yellow = "\\x1B[33;21m"
    red = "\\x1B[31;21m"
    bold_red = "\\x1b[31;1m"
    reset = "\\x1B[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def logging_config(logging, logfile=None, lineno=True, funcname=True, classname=True, level=1):
    """  """
    if classname:
        logging.setLogRecordFactory(logging_LogRecord)

    stdout_handler = colorlog.StreamHandler()
    color_formatter = colorlog.ColoredFormatter(
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        # datefmt=None, %(module)s %(name)s
        "%(log_color)s%(levelname)s%(reset)s %(cyan)s%(asctime)s%(reset)s" + \
               ("%(yellow)s %(module)s.%(funcName)s" if funcname else "") + \
               (".%(lineno)d" if lineno else "") + \
               ": %(reset)s%(message)s",
        datefmt='%m/%d %H:%M:%S',  # %Y-%m-%d
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%',
    )

    # color_formatter = ColorFormatter()
    stdout_handler.setFormatter(color_formatter)
    
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(color_formatter)

    logging.basicConfig(
        format="%(levelname)s %(asctime)s" + \
               (" @%(module)s.%(funcName)s" if funcname else "") + \
               (" [%(lineno)4d]" if lineno else "") + \
               ":: %(message)s",
        datefmt='%m/%d %H:%M:%S',  # %Y-%m-%d
        handlers=[stdout_handler, file_handler] if logfile else [stdout_handler],
        )
    # getLogger() returns the root logger
    logging.getLogger().setLevel([logging.WARNING, logging.INFO, logging.DEBUG][level])


def logger_setlogfile(logfile, level=1, logger=None, replace_all=True):
    """ set new log file """

    if replace_all:
        loggers = [logging.getLogger()] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    else:
        loggers = [logging.getLogger() if logger is None else logger]

    # remove existing handlers
    for _logger in loggers:
        for _handler in _logger.handlers:
            if isinstance(_handler, logging.FileHandler):
                _logger.removeHandler(_handler)
            # _logger = _logger.parent

    # add new handler to the top logger only
    new_handler = logging.FileHandler(logfile)
    new_handler.setFormatter(logging.root.handlers[0].formatter)
    new_handler.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][level])
    loggers[0].addHandler(new_handler)
    return new_handler


def logger_setlevel(logger, level=1):
    """  """
    logger.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][level])


def parser_formatter():
    return lambda prog: argparse.RawTextHelpFormatter(prog, indent_increment=1, max_help_position=42, width=200)


def parser_flex_var(name, argv=''):
    """ allow - or -- for optioal var; allow - or _ in varnames for flexibility """
    name = name.lstrip('-')
    if '-h' in argv or '--help' in argv:
        return [f'-{name}']
    else:
        if '-' in name:
            name = [name, name.replace('-', '_')]
        elif '_' in name:
            name = [name, name.replace('_', '-')]
        else:
            name = [name]
        return [f'-{_name}' for _name in name] + [f'--{_name}' for _name in name]


def parser_rmv_arguments(parser, args):
    """ remove positional and optional argument from parser """
    for name in args:
        name = name.lstrip('-')
        for action in parser._actions:
            if action.dest == name:
                parser._remove_action(action)
                break
    
    
def parser_rmv_options(parser, options):
    for option in options:
        for action in parser._actions:
            if action.option_strings and option in action.option_strings:
                parser._remove_action(action)
                break
    

def argv_argparse(parser, argv):
    """ essentially run parser.parse_args(argv) with three additions:
        1) unpack the argv if it is a nested list/tuple
        2) add argv to args (disabled)
        3) add unknown args as args.jitargs
        4) return Struct()
    """
    if isinstance(argv, str): argv = [argv]
    argv = unpack_list_tuple(argv)
    # add - to optional arguments in argv if only has one leading -
    argv = [f'-{_s}' if _s.startswith('-') and not _s.startswith('--') and not str_isnumeric(_s) else _s for _s in argv]

    args, unknown = parser.parse_known_args(argv)
    args = Struct(vars(args))
    if len(unknown):
        logger.info(f'Parsing undefined args: {unknown}')
        jitargs = argv_myparse(unknown)[1].__dict__
        args.update(jitargs=jitargs)
    else:
        args.update(jitargs={})
    # args.argv = " ".join(argv)
    return args


def argv_myparse(argv, **kwargs):
    """ home-made argv parser, return (positional_args, kwargs as Struct()) """
    if isinstance(argv, str):
        argv = [argv]
    argv = [str(_s) for _s in argv]

    psargs = []
    while len(argv):
        arg = argv.pop(0)
        if arg.startswith('-') and not str_isnumeric(arg, finite=True):   # kwargs
            # arg = re.sub("^-*", "", arg)
            # strip leading - and --
            arg = arg.lstrip('-')
            if '=' in arg:
                arg = arg.split('=')
                kwargs[arg[0].replace('-', '_')] = str_auto_value('='.join(arg[1:]))
            else:
                key = arg.replace('-', '_')
                kwargs[key] = []

                while len(argv):
                    if argv[0].startswith('-') and not str_isnumeric(argv[0], finite=True):
                        break
                    kwargs[key].append(str_auto_value(argv.pop(0)))
                    
                if len(kwargs[key]) == 0:
                    kwargs[key] = True
                elif len(kwargs[key]) == 1:
                    kwargs[key] = kwargs[key][0]
        else: # position args
            psargs.append(str_auto_value(arg))

    kwargs.setdefault('verbose', kwargs.get('v', 1))

    if len(psargs) == 1:
        psargs = psargs[0]

    return psargs, Struct(kwargs)


def argv_fn_caller(argv, module=None, **kwargs):
    """ call functions in the module based on argv (sys.argv[1:]) """
    # parse the inputs
    psargs, args = argv_myparse(argv, **kwargs)

    # set up verbosity and logging
    if not hasattr(args, 'verbose'):
        if hasattr(args, 'v'):
            args.verbose = args.v
        else:
            args.verbose = 1
    if hasattr(args, 'v'): delattr(args, 'v')

    logging_config(logging, logfile=args.__dict__.get('logfile', None), 
        lineno=True, level=args.verbose)

    # prepare args
    if isinstance(psargs, str):
        psargs = [psargs]

    if module is None: # get the caller module
        module = inspect.getmodule(inspect.currentframe().f_back)
        # module = inspect.getmodule(inspect.stack()[-1].frame) # this fails in python debugger

    # get the list of functions defined locally in the "module"
    module_local_fns = inspect.getmembers(module,
        predicate = lambda f: inspect.isfunction(f) and f.__module__ == module.__name__)
    func_names, func_callers = list(zip(*module_local_fns))

    # no subcommand is passed
    if len(psargs) == 0: #  and ('-h' in sys.argv or '--help' in sys.argv:
        print(f'\nUsage: {str_color(Path(module.__file__).name, style="normal")} mission [args]\n')
        print('  Mission is the first positional argument and the rest will be passed to mission()!\n')
        print('  Mission can be a quoted regex to run multiple missions with the SAME args.')
        print('  The regex will be capped with ^ and $ to match the FULL mission name.\n')
        print('  Pass -h or --help to get help on the mission\n')
        print('  Available missions:')
        
        max_strlen = max([26] + [len(_name) for _name in func_names])
        for func_name, func_caller in module_local_fns:
            if func_name.startswith('_'): continue
            help_txt = f'     {func_name:{max_strlen}s} - ' + \
                (func_caller.__doc__.split("\n")[0] if func_caller.__doc__ else '')
            print(help_txt)
        exit(0)

    num_run_actions = 0
    action_pattern = f'^{psargs[0]}$'
    for ifunc, fn_name in enumerate(func_names):
    # if action_pattern in func_names:
        # ifunc = func_names.index(action_pattern)
        if not re.match(action_pattern, fn_name):
            continue

        if '-h' in argv or '--help' in argv:
            print(f'docstring: {str_color(func_callers[ifunc].__doc__.strip() if func_callers[ifunc].__doc__ is not None else "null", style="normal", color="green")}')
            # source_codes = inspect.getsourcelines(func_callers[ifunc])[0]
            # for line in source_codes:
            #     print(line.rstrip())
            #     if line.rstrip().endswith(':') and not line.lstrip().startswith('#'):
            #         break
            # subprocess.run(["sed", "-n", f"'/^def {fn_name}/,/^def /p'", module.__file__, "|" f"\grep --color=always '#.*#$'"])
            os.system(f"sed -n '/^def {fn_name}(/,/):$/p' {module.__file__} | highlight -O xterm256 --syntax=python --force") # | \column -t -s# -n")
            print("\nAvailable options (between paired #s):\n")
            os.system(f"sed -n '/^def {fn_name}(/,/^def /p' {module.__file__} | \grep '#.*#$' | highlight -O xterm256 --syntax=python --force | \column -t -s# -n")
            # exit(0)
        else:
            logger.info(f'Executing mission: {str_color(fn_name)}...')
            if len(psargs) > 1:
                func_callers[ifunc](*psargs[1:], 
                    __date__=time_now_str(),
                    __name__=f'{Path(module.__file__).stem}.{fn_name}', 
                    __cwd__=Path.cwd().resolve().as_posix(),
                    **vars(args))
            else:
                func_callers[ifunc](
                    __date__=time_now_str(),
                    __name__=f'{Path(module.__file__).stem}.{fn_name}', 
                    __cwd__=Path.cwd().resolve().as_posix(), 
                    **vars(args))

        num_run_actions += 1

    if num_run_actions == 0:
        logger.error(f'Cannot recognize mission: {action_pattern} in {module.__file__}!')
    else:
        logger.info(f"Successfully run {num_run_actions} mission(s) with pattern: {action_pattern}!")
    return


def argv_lookup_values(argv, args=Struct()):
    """ generate a struct with all optional arguments in argv.
        If a key exists in args, its value is copied.
    Caution: it only look for --OPTION or -OPTION """
    keys = [_key.replace('-', '') for _key in argv if _key.startswith(('--', '-'))]
    cline_args = dict.fromkeys(keys)
    for _key in set(keys).intersection(vars(args).keys()) :
        cline_args[_key] = vars(args)[_key]
    return Struct(**cline_args)


def time_elapsed():
    pass


def time_now_str(fmt="%I:%M%p %B %d, %Y"):
    return datetime.now().strftime(fmt)


def round_sig(x, sig=2):
    """ round to the given number of significant digits  """
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def alphanum_start(str_in):
    """ find the longest continuous alphabet/number at the beginning of a string """
    i = re.search(r'\W+', str_in).start()
    return str_in[0:i]


def alphanum_first(str_in):
    """ find the first continuous alphabet/number at the beginning of a string """
    res = re.search(r'\w+', str_in)
    return str_in[res.start():res.end()]


def numbers_first(str_in):
    """ find the first continuous alphabet/number at the beginning of a string """
    res = re.search(r'\d+', str_in)
    return str_in[res.start():res.end()]


def num2str(num, sig=3):
    num = round_sig(num, sig)
    sign = ''

    metric = {'T': 1000000000000, 'B': 1000000000, 'M': 1000000, 'K': 1000, '': 1}

    for index in metric:
        num_check = num / metric[index]

        if(num_check >= 1):
            num = num_check
            sign = index
            break

    return f"{str(num).rstrip('0').rstrip('.')}{sign}"


def fix_length(data, length=7, skip_dims=tuple(), align='left', **kwargs):
    """ cut or pad to the given length, use """
    if not hasattr(data, '__len__'): 
        data = np.array([data])
    if skip_dims is None:
        skip_dims = []
    elif not hasattr(skip_dims, '__len__'):
        skip_dims = [skip_dims]
    if not hasattr(length, '__len__'):
        length = [length] * data.ndim

    pad_width = ()
    pad_noyes = False
    for i, dim in enumerate(data.shape):
        if i in skip_dims:
            pad_width += (0, 0),
            continue
        if dim > length[i]:
            data = data.take(indices=range(length[i]), axis=i)
            pad_width += (0, 0),
        elif dim == length[i]:
            pad_width += (0, 0),
        else:
            pad_width += ((0, length[i] - dim),) if align == 'left' else ((length[i] - dim, 0),)
            pad_noyes = True

    if pad_noyes:
        return np.pad(data, pad_width, **kwargs)
    else:
        return data


def fix_length1d(data, length, **kwargs):
    """ if needed, np.pad is used for padding with the same kwargs """
    if not hasattr(data, '__len__'): data = np.aray([data])
    data_len = len(data)

    if data_len >= length:
        return data[:length]
    else:
        return np.pad(data, (0, length - data_len), **kwargs)


def fix_length2d(data, length, **kwargs):
    """ data is 2D matrix without batch dim
    np.pad is used for padding with **kwargs"""
    data_len = data.shape

    if isinstance(length, int) or isinstance(length, np.integer):
        length = [length]

    len2pad = [0, 0]

    if data_len[0] >= length[0]: # check 1st dimension
        data = data[:length[0], :]
    else:
        len2pad[0] = length[0] - data_len[0]

    if data_len[1] >= length[-1]: # check 2nd dimension
        data = data[:, :length[-1]]
    else:
        len2pad[1] = length[-1] - data_len[1]

    if any(len2pad): # pad if needed
        return np.pad(data, ((0, len2pad[0]), (0, len2pad[1])), **kwargs)
    else:
        return data
        

def get_list_index(vlist, v, offset=0, last=False):
    # get_index = lambda vlist, v : [vlist.index(v)] if v in vlist else []
    try:
        if last:
            v_reversed = vlist[-1::-1]  # vlist.reverse() changes vlist
            return [len(v_reversed) - 1 - v_reversed.index(v) + offset]
        else:
            return [vlist.index(v) + offset]
    except ValueError:
        return []


def unpack_list_tuple(*list_in, types2unpack=(list, tuple)):
    if len(list_in) == 0:
        return []
    list_out = []
    for _item in list_in:
        if type(_item) in types2unpack:
            list_out.extend(unpack_list_tuple(*_item))
        else:
            list_out.append(_item)
    return list_out


def counter_from_keyvals(keys, vals, sep=None):
    """ parse string pairs such as 'True|False' and '80|20' to {'True': 80, 'False': 20} """
    if len(keys) == 0 or len(vals) == 0:
        return Counter()
    if sep:
        return Counter(dict(zip(keys.split(sep), map(int, vals.split(sep)))))
    else:
        return Counter({keys: int(vals)})


def fuzzy_name(str_in):
    """ convert to lower cases and remove common delimiters """
    if isinstance(str_in, str):
        return str_in.lower().replace('_', '').replace('-', '')
    else:
        return [_s.lower().replace('_', '').replace('-', '')
                if isinstance(_s, str) else _s for _s in str_in]


def zflatten2xyz(z, x=None, y=None):
    """ flatten an nxm 2D array to [x, y, z] of shape=(n*m, 3)"""
    if x is None:
        x = np.arrange(0, z.shape[0], step=1)
    if y is None:
        y = np.arrange(0, z.shape[1], step=1)
    xlen = len(x)
    ylen = len(y)
    assert z.shape[0] == xlen and z.shape[1] == ylen, 'check dimensions!!!'

    xx, yy = np.meshgrid(x, y)
    xx = xx.T
    yy = yy.T  # meshgrid take the second dimension as x

    xylen = xlen*ylen
    return np.concatenate((xx.reshape((xylen, 1)),
                           yy.reshape((xylen, 1)),
                           z.reshape((xylen, 1))), axis=1)


def zflatten2xyz_debug():
    x = np.arange(0, 5, step=1)
    y = np.arange(0, 4, step=1)

    xx, yy = np.meshgrid(x, y)

    z = xx.T*10 + yy.T
    print('z[2,3]:]', z[2, 3])
    print('x', x)
    print('y', y)
    print('z', z)
    xyz = zflatten2xyz(z, x=x, y=y)
    print(xyz)


def send_email(
    sender='python.script@localhost',
    receivers=['xiangyun@gmail.com'],
    subject="You've Got Mail",
    content="\n".join(["This is a test message.", "Please ignore!"]),
    context_type='plain',
    server='localhost',
    attachments=None,
    **kwargs):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.utils import format_datetime

    msg = MIMEMultipart()
    # msg['Date'] = format_datetime(localtime=True)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(receivers) if isinstance(receivers, list) else receivers
    msg.attach(MIMEText(content, context_type))

    for f in attachments or []:
        fname = os.path.basename(f)
        with open(f, "rb") as iofile:
            part = MIMEApplication(iofile.read(), Name=fname)
        part['Content-Disposition'] = 'attachment; filename="%s"' % fname
        msg.attach(part)

    try:
        smtp_obj = smtplib.SMTP(server)
        smtp_obj.sendmail(sender, receivers, msg.as_string())         
        print("Successfully sent email")
    except smtplib.SMTPException:
        print("Error: unable to send email")
    finally:
        smtp_obj.close()


if __name__ == '__main__':
    zflatten2xyz_debug()
