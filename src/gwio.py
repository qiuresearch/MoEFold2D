#!/usr/bin/env python
import bz2
import functools
import glob
import gzip
import json
import logging
import os
import pickle
import re
import sys
# import lzma
# import brotli
from pathlib import Path

import numpy as np
import pandas as pd

# homebrew
import misc

logger = logging.getLogger(__name__)

def list2str(list_in, spacing=2, screen_width=None, sort=False):
    """ format a list of strings to be shown in a terminal """

    if screen_width is None:
        try:  # it may fail under screen environment
            screen_width = int(os.get_terminal_size().columns * 0.92)
        except:
            screen_width = 80

    half_width_p4 = screen_width / 2 + 2 + spacing # include quoting ""

    if sort:
        list_in = sorted(list_in)
    list_lens = [len(_s) for _s in list_in]

    min_len = min(list_lens)
    if min_len > half_width_p4:
        return '\n'.join(list_in) + '\n'

    # long_list: list of long strings that wil take one full line (or more)
    # short_list: list of short strings, multiples of which can fit in one line
    max_len = max(list_lens)
    if max_len > half_width_p4:
        long_list, short_list = [], []
        # min_len_long = max_len
        # max_len_short = min_len
        max_len = min_len
        for i, slen in enumerate(list_lens):
            if slen > half_width_p4:
                long_list.append(list_in[i])
                # if slen < min_len_long:
                #     min_len_long = slen
            else:
                short_list.append(list_in[i])
                if slen > max_len:
                    max_len = slen
        list_in = short_list            
    else:
        long_list = None

    num_per_line = screen_width // (max_len + 4)
    str_width = (screen_width // num_per_line) - 2

    fmt_strs = []
    for i, s in enumerate(list_in):
        fmt_strs.append(' ' * (str_width - len(s)) + "'{:s}'" + 
            ('\n' if (i+1) % num_per_line == 0 else ""))

    fmt_str = ''.join(fmt_strs)
    # fmt_str = ' '.join([fmt_str] * min([len(list_in), num_per_line])) + '\n'
    # print(fmt_str)
    str_out = fmt_str.format(*list_in)
    if long_list is None:
        return str_out
    else:
        return str_out + '\n'.join(long_list) + '\n'


def str2filename(str_in,
                 trans_table=str.maketrans(
                    '()[]:/;,\n',    # these are the characters to be replaced
                    '-----____',     # these are the replacements
                    '\\'), # remove these characters
                ):
    return misc.str_deblank(str_in.translate(trans_table))


def last_backup_path(fname):
    """ fname is returned if not backup path exists """
    if isinstance(fname, str): fname = Path(fname)

    if fname.exists():
        all_backups = list(fname.parent.glob(fname.name + '.[0-9]*'))
        if len(all_backups):
            suffixes = np.array([int(_f.suffix[1:]) for _f in all_backups])
            ilast = suffixes.argmax()

            fname = all_backups[ilast]

    return fname


def new_path_with_backup(fname):
    """  """
    if isinstance(fname, str): 
        fname = Path(fname)

    if fname.exists(): # backup file
        last_file = last_backup_path(fname)

        if last_file.samefile(fname): # no backup exists
            new_suffix = '.001'
        else:
            new_suffix = f'.{int(last_file.suffix[1:]) + 1:03d}'

        new_file = Path(fname.as_posix() + new_suffix)
        if fname.is_file():
            logger.info(f'Backing up file: {misc.str_color(fname)} to: {misc.str_color(new_file)}')
            fname.rename(new_file)
        else:
            logger.info(f'Backing up directory: {misc.str_color(fname)} to: {misc.str_color(new_file)}')
            fname.rename(new_file)

    return fname


def copy_text_file_to_dir(src_file, save_dir, overwrite=False):
    """ save text file to a directory """
    if isinstance(src_file, str): src_file = Path(src_file)
    if isinstance(save_dir, str): save_dir = Path(save_dir)

    des_file = save_dir / src_file.name
    if des_file.exists() and src_file.samefile(des_file):
        logger.info(f'No need to copy text file (the same path!): {misc.str_color(src_file)}')
        return des_file

    if des_file.exists() and not overwrite:
        des_file = new_path_with_backup(des_file)
        logger.info(f'Text file: {src_file} aleady exists, saving as: {misc.str_color(des_file)}')
    else:
        logger.info(f'Saving text file to: {misc.str_color(des_file)}')

    des_file.write_text(src_file.read_text())
    return des_file


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def pickle_squeeze(data, pkl_file, fmt='lzma'):
    """ looks like compress_pickle may be the way to go """
    if isinstance(pkl_file, str): pkl_file = Path(pkl_file)

    if fmt is None: fmt = []
    if type(fmt) not in (list, tuple): fmt = [fmt]
    if len(fmt): fmt = [_s.lower() for _s in fmt]

    with pkl_file.open('wb') as hfile:
        logger.debug(f'Storing pickle: {pkl_file}')
        pickle.dump(data, hfile, -1)

    pkl_fname = pkl_file.as_posix()
    if 'gz' in fmt or 'gzip' in fmt:
        zip_file = pkl_fname + '.gz'
        logger.debug(f'Zipping and storing pickle: {zip_file}')
        with gzip.open(zip_file, 'wb') as hfile:
            pickle.dump(data, hfile, -1)

    if 'bz' in fmt or 'bz2' in fmt:
        zip_file = pkl_fname + '.pbz2'
        logger.debug(f'Zipping and storing pickle: {zip_file}')
        with bz2.BZ2File(zip_file, 'wb') as hfile:
            pickle.dump(data, hfile, -1)

    # if 'lzma' in fmt:
    #     zip_file = pkl_fname + '.xz'
    #     logger.debug(f'Zipping and storing pickle: {zip_file}')
    #     with lzma.open(zip_file, 'wb') as hfile:
    #         pickle.dump(data, hfile, -1)

    # with open('no_compression.pickle', 'rb') as f:
    #     pdata = f.read()
    #     with open('brotli_test.bt', 'wb') as b:
    #         b.write(brotli.compress(pdata))


def dict2json(dict_in, fname='args.json', fdir=None, indent=(4, None), overwrite=False):
    """ if indent is a list, customized json encoder is used """
    if fdir is not None:
        fdir = Path(fdir)
        fdir.mkdir(parents=True, exist_ok=True)
        json_file = fdir / fname
    else:
        json_file = Path(fname)
        
    if json_file.exists() and not overwrite:
        json_file = new_path_with_backup(json_file)
        
    if type(indent) in (list, tuple):
        with open(json_file, 'w') as hfile:
            hfile.writelines(json_str(dict_in, indent=indent[0]))
        return json_file

    # serialize or remove keys from dict_in
    dict_bkp = {} # back up
    keys_rmv = [] # only store names
    for key, val in dict_in.items():
        if not is_jsonable(val):
            dict_bkp[key] = val # save the original

            if isinstance(val, Path):
                dict_in[key] = val.as_posix()
            else:
                keys_rmv.append(key)
    if len(keys_rmv):
        logger.info(f'The following keys are not jsonable: {keys_rmv}')
        for key in keys_rmv: dict_in.pop(key)

    logger.info(f'Saving json file: {misc.str_color(json_file)}...')

    with open(json_file, 'w') as hfile:
        if len(keys_rmv):
            pass
            # logger.warning(f'The following keys are excluded from JSON: {keys_rmv}')
            # hfile.write(f'# Keys not jsonable: {keys_rmv}\n')
        # if indent and oneline_list:
        #     out_list = json.dumps(dict_in, indent=indent)
        #     out_list = re.sub(r'([\t\f\v\r\n]+)([^:\[\]]+),[\s\$]+', r'\1\2, ', out_list)
        #     out_list = re.sub(r'([\t\f\v\r\n]+)([^:\[\]]+),[^[:print:]]+', r'\1\2, ', out_list)
        #     # out_list = re.sub(r'([\t\f\v\r\n]+)\s+([^:\[\]]+),\s+', r'\1\2, ', out_list)
        #     # out_list = re.sub(r'([^[:print:]]+)\s+([^:\[\]]+),\s+', r'\1\2, ', out_list)
        #     out_list = re.sub(r'": \[\s+', '": [', out_list)
        #     # out_list = re.sub(r'",\s+', '", ', out_list)
        #     out_list = re.sub(r'\s+\],', '],', out_list)
        #     hfile.writelines(out_list)
        # else:
        json.dump(dict_in, hfile, indent=indent)

    dict_in.update(dict_bkp) # restore backup
    return json_file

def json_str(val_in, max_size=42, fmt=None, indent=3, sep=[', ', ': '], space=' ', newline='\n', level=1):
    """ convert a dict to json string
    adopted from https://stackoverflow.com/questions/10097477/python-json-array-newlines 
        max_size: maximum number of elements in a list or array to be shown
    """

    INDENT = indent if indent else 0
    SPACE = space if space else ''
    NEWLINE = newline if newline else ''
    SEP = sep if sep else ''

    def to_json(o, level=1, ):
        if isinstance(o, dict) or hasattr(o, 'items'):
            ret = ["{"]
            key_indent = SPACE * INDENT * level
            for k,v in o.items():
                ret.append(f'{key_indent}"{str(k)}"{SEP[-1]}{to_json(v, level + 1)},')
            if len(ret) > 1:
                ret[-1] = ret[-1][:-1]
            ret.append(SPACE * INDENT * (level-1) + "}")
            ret = NEWLINE.join(ret)
        elif isinstance(o, str):
            ret = f'"{o }"'
        elif isinstance(o, list) or isinstance(o, tuple):
            ret = f"[{SEP[0].join([to_json(e, level+1) for e in o])}]"
        elif isinstance(o, bool):
            ret = "true" if o else "false"
        elif isinstance(o, int) or isinstance(o, np.integer):
            ret = str(o)
        elif isinstance(o, float): # or isinstance(o, np.float):
            ret = f'{o:.7g}'
        elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
            if o.size <= max_size:
                ret = f"[{SEP[0].join(map(str, o.flatten().tolist()))}]"
            else:
                ret = f'[{o.shape[0]}x{o.shape[1]} np.integer array]'
        elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
            if o.size <= max_size:
                ret = f"[{SEP[0].join(map(lambda x: f'{x:.7g}', o.flatten().tolist()))}]"
            else:
                ret = f'[{o.shape[0]}x{o.shape[1]} np.inexact array]'
        elif isinstance(o, Path):
            ret = f'"{o.as_posix()}"'
        elif isinstance(o, functools.partial):
            ret = f'"{o.func}"' # str(o) gives all parameters
        elif type(o) in ('module', 'function'):
            ret = f'"{str(o)}"'
        elif o is None:
            ret = 'null'
        elif isinstance(o, pd.DataFrame) or isinstance(o, pd.Series):
            ret = str(type(o))
        else:
            # raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
            logger.debug(f'Unknown type {str(type(o))} for json serialization!')
            # print(o)
            ret = 'UNK'
        return ret

    if isinstance(val_in, dict):
        return to_json(val_in, level=level)
    elif hasattr(val_in, '__dict__'):
        return to_json(val_in.__dict__, level=level)
    else:
        logger.warning(f'May not be able to jsonize type: {type(val_in)}, will try...')
        return to_json(val_in, level=level)


def json2dict(fname='args.json', fdir=None):
    """  """
    if fdir is not None:
        fname = Path(fdir) / fname

    # logger.info(f'Loading json file: {json_file}...')
    with open(fname, 'r') as hfile:
        dict_out = json.load(hfile)

    return dict_out


def files_check_sibling(in_files, fdir=None, suffix=None):
    """ check whether the same fname.stem in fdir with suffix exists """
    num_missed = 0
    if fdir is None:
        for fname in in_files:
            fname = Path(fname)
            if not fname.with_suffix(suffix).exists():
                num_missed += 1
                print(f'No {suffix} file found for: {fname.as_posix()}')
    else:
        fdir = Path(fdir)
        for fname in in_files:
            fname = Path(fname)
            if not (fdir / (fname.stem + suffix)).exists():
                num_missed += 1
                print(f'No {suffix} file found for: {fname.as_posix()}')

    if num_missed == 0:
        logger.info(f'Found {suffix} for every file, yahoo!')
    else:
        logger.info(f'Number of files without {suffix}: {num_missed}')
    return num_missed


def get_file_lines(fname, fdir='', strip=False, keep_empty=True,
                   comment='#', keep_comment=True):
    """  """
    flines = []
    if type(fname) not in (list, tuple): fname = [fname]
    for onefile in fname:
        with open(os.path.join(fdir, onefile), 'r') as iofile:
            if strip:
                flines.extend([_s.strip() for _s in iofile.readlines()])
            else:
                flines.extend(iofile.readlines())

    if keep_comment and keep_empty:
        return flines
    elif keep_comment:
        return [_s for _s in flines if len(_s) > 0]
    elif keep_empty:
        return [_s for _s in flines if not _s.startswith(comment)]
    else:
        return [_s for _s in flines if len(_s) > 0 and _s[0] != comment]


def get_dataframes(inputs, fmt='infer', sep=',', header='infer', skiprows=0,
            concat=True, axis=0, sort=False, ignore_index=True, 
            keep_cols=None, drop_cols=None, keep_rows=None, drop_rows=None,
            return_files=False, return_save_prefix=False):
    """ read in dataframes from files or variables of various formats """
    if type(inputs) not in (list, tuple): inputs =[inputs]

    # glob file names
    input_list = []
    for input in inputs:
        if isinstance(input, str):
            input_list.extend(glob.glob(input))
        elif isinstance(input, Path):
            input_list.extend(glob.glob(input.as_posix()))
        else:
            input_list.append(input)

    num_inputs = len(input_list)
    dfs, input_names = [], []
    for i, input in enumerate(input_list):
        # determine df_fmt, sep, etc.
        if fmt == 'infer':
            if not isinstance(input, str): # it should be a variable
                suffix = 'var'
            elif input.endswith(('.gz', '.zip')):
                suffix = input.split('.')[-2].lower()
            else:
                suffix = input.split('.')[-1].lower()

            if suffix == 'var':
                if isinstance(input, pd.DataFrame):
                    df_fmt = 'dataframe'
                elif isinstance(input, pd.DataFrame):
                    df_fmt = 'dict'
                elif isinstance(input, pd.Series):
                    df_fmt = 'dataseries'
                elif isinstance(input, np.ndarray):
                    df_fmt = 'np.ndarray'
                else:
                    df_fmt = 'UNK'
                    logger.error(f'Cannot infer format for variable type: {type(input)}!!!')
            elif suffix == 'csv':
                sep = ','
                df_fmt = 'csv'
            elif suffix == 'tsv':
                sep = '\t'
                df_fmt = 'csv'
            elif suffix in ['txt', 'upp', 'dat']:
                header = None
                sep = r'\s+'
                df_fmt = 'csv'
            elif suffix in ['pkl', 'pickle']:
                df_fmt = 'pkl'
            elif suffix == 'json':
                df_fmt = 'json'
            elif '.pkl' in input or '.pickle' in input:
                df_fmt = 'pkl'
            elif '.csv' in input:
                seq = ','
                df_fmt = 'csv'
            else:
                df_fmt = 'csv'
                logger.error(f'Cannot infer format for suffix: {misc.str_color(suffix, color="red")}!!!')
        else:
            suffix = ''
            df_fmt = fmt

        # determine input_name
        if suffix == 'var':
            input_names.append('var')
            logger.info(f"Appending variable fmt: {df_fmt} ({i + 1}/{num_inputs})")
        else:
            input_names.append(input)
            logger.info(f"Reading file: {misc.str_color(input)} of fmt: {df_fmt} ({i + 1}/{num_inputs})")

        # read data and append
        df_fmt = df_fmt.lower()
        if df_fmt == 'dataframe':
            dfs.append(input)
        elif df_fmt == 'dict':
            dfs.append(pd.DataFrame(input))
        elif df_fmt == 'dataseries':
            dfs.append(pd.DataFrame(input))
        elif df_fmt == 'np.ndarray':
            dfs.append(pd.DataFrame(input))
        elif df_fmt in ['csv', 'tsv', 'txt']:
            dfs.append(pd.read_csv(input, sep=sep, header=header, skiprows=skiprows))
        elif df_fmt == 'json':
            dfs.append(pd.read_json(input))
        elif df_fmt == 'pkl':
            try:
                pkldata = pd.read_pickle(input)
            except:
                with open(input, 'rb') as iofile:
                    pkldata = pickle.load(iofile)
            if isinstance(pkldata, dict):
                pkldata = pd.DataFrame(pkldata)
            dfs.append(pkldata)
        else:
            logger.error(f'Unsupported fmt: {df_fmt}!!!')
        
        cols_desc = dfs[-1].columns.astype(str).to_list() if len(dfs[-1].columns) < 6 else \
            dfs[-1].columns[0:3].astype(str).to_list() + ['...'] + dfs[-1].columns[-2:].astype(str).to_list()
        logger.info(f'Dataframe shape: {dfs[-1].shape}, cols: [{", ".join(cols_desc)}]')

        if keep_cols is not None:
            dfs[-1].drop(columns=dfs[-1].columns.difference(keep_cols), inplace=True, errors='ignore')
        if drop_cols is not None:
            dfs[-1].drop(columns=drop_cols, inplace=True, errors='ignore')
        if keep_rows is not None:
            dfs[-1].drop(index=dfs[-1].index.difference(keep_rows), inplace=True, errors='ignore')
        if drop_rows is not None:
            dfs[-1].drop(index=drop_rows, inplace=True, errors='ignore')
        
        if logger.root.level < logging.INFO:
            logger.info(f'All columns are:')
            print(list2str(dfs[-1].columns.to_list(), sort=False))

    if concat and len(dfs):
        if len(dfs) > 1:
            df_out =  pd.concat(dfs, axis=axis, sort=sort, ignore_index=ignore_index)
            logger.info(f'The combined dataframe has shape: {df_out.shape}')
        else:
            df_out = dfs[0]

        if any(df_out.columns.duplicated()):
            logger.warning('Dataframe contains duplicated column names!!!')
            # df_out.columns = list(range(df_out.shape[1]))
    elif len(dfs):
        df_out = dfs
    else:
        df_out = None
        logger.error(f'Files {inputs} not found or unreadable!!!')

    if return_files:
        return df_out, input_names
    elif return_save_prefix:
        return df_out, '-'.join([Path(_f).stem for _f in input_names])
    else:
        return df_out


def setcwd():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    print('Current working directory: ' + os.getcwd())
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print('Changing to the file directory: ' + dname)
    os.chdir(dname)


def showinfo(infostr="checking showinfo", infotype='info'):
    """Display information during execution of a function. The name of
    the function will be shown!

    infostr  -- the information string
    infotype -- the kind of information: info, warning, error

    Return None
    """
    from traceback import extract_stack
    callerinfo = extract_stack()[-2]
    # print extract_stack()
    if callerinfo[2] == '<module>':
        callername = os.path.basename(callerinfo[0])
        if callername[0] == '<':
            callername = 'gwio'
    else:
        callername = callerinfo[2]
    print("[%s::%s] %s" % (infotype.upper(), callername, infostr))
    return

if __name__ == '__main__':
    misc.argv_fn_caller(sys.argv[1:]) # module=sys.modules[__name__], verbose=1)
