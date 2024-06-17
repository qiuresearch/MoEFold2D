#!/usr/bin/env python
import inspect
import functools
import itertools
import json
import logging
import math
import os
import pickle
import sys
from collections import Counter, namedtuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import cluster, metrics, manifold
from tqdm import tqdm

# homebrew
import gwio
import misc
import molstru
from molstru import reslet_lookup, seq_fuzzy_match

logger = logging.getLogger(__name__)

def get_midat(inputs,
              data_dir=None,
              return_save_prefix=False,
              info=False,
              header='infer',
              **kwargs):
    """ a flexible reader of midat """

    save_prefix = None

    if isinstance(inputs, pd.DataFrame) or isinstance(inputs, pd.Series):
        df = inputs
        save_prefix = 'var_pd'
    elif isinstance(inputs, dict):
        df = pd.DataFrame(inputs, copy=False)
        save_prefix = 'var_dict'
    else: # file names
        if type(inputs) not in (list, tuple):
            inputs = [inputs]

        if data_dir is None:
            inputs = [Path(_f) for _f in inputs if _f is not None]
        else:
            inputs = [Path(data_dir) / _f for _f in inputs if _f is not None]

        if len(inputs):
            if inputs[0].suffix.endswith(('.fasta', '.fa')):
                df = molstru.SeqStruct2D(inputs, fmt='fasta').to_df(
                        seq=True, dbn=False, ct=False, has=False, res_count=False)
            elif inputs[0].suffix.endswith('dbn'):
                df = molstru.SeqStruct2D(inputs, fmt='dbn').to_df(
                        seq=True, dbn=True, ct=False, has=False, res_count=False)            
            else:
                df = gwio.get_dataframes(inputs, fmt='infer', return_files=False,
                        ignore_index=True, concat=True, header=header)

            save_prefix = '-'.join([_f.stem for _f in inputs])
        else:
            df, save_prefix = None, None
            
    logger.debug(('' if save_prefix is None else f'File: {save_prefix}, ') \
                 + "INVALID or NONEXISTING!!!" if df is None else f'midat shape: {df.shape}')
    if info:
        df.info()
        print(df.describe())

    if return_save_prefix:
        return df, save_prefix
    else:
        return df


def extract_rnacentral_stRNA(data_names='ncRNA.pkl', save_prefix=None, **kwargs):
    """  """
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix + '_stRNA', 'autosave'])

    mol_types = [
        'rRNA', 'tRNA', 'lncRNA', # 'miscRNA',
        'sRNA', # bacterial small RNAs (50-500 nts), highly structured
        'snoRNA', 'snRNA', # 'piRNA',
        'hammerhead_ribozyme',
        'self_splicing_intron',
        'SRP', 'tmRNA', 'RNaseP', 'ribozyme',
        'Y_RNA', # components of the Ro60 ribonucleoprotein particle
        'RNaseMRP',
        'vRNA', 'scRNA', 'telomerase',
        'scaRNA',
    ]

    df = df[df['moltype'].isin(mol_types)]

    # logger.info(f'Saving new df {df.shape} to pickle file: {save_prefix}.pkl ...')
    # df.to_pickle(save_prefix + '.pkl')
    save_lumpsum_files(df, save_prefix=save_prefix, save_pkl=True, save_fasta=True)


def count_all_residues(
        data_names,
        args=misc.Struct(),
        prefix='',
        show=True,
        tqdm_disable=False,
        **kwargs):
    """ count the residue names in ALL rows and report statistics """

    df = get_midat(data_names)

    res_count = Counter()
    for seq in df.seq:
        res_count.update(Counter(seq))

    total_count = float(sum(res_count.values()))
    for key, val in res_count.most_common():
        args.update({f'{prefix}num_{key}': val})
        args.update({f'{prefix}pct_{key}': val / total_count})

    args.update({f'{prefix}total': total_count})

    if show: print(gwio.json_str(args.__dict__))


def count_ct_bps(
        data_names,
        recap=misc.Struct(),
        prefix='',
        show=True,
        plot=False,
        tqdm_disable=False,
        **kwargs):
    """ count the paired bases in pkldata and report statistics """

    df = get_midat(data_names)

    bp_counter = Counter()
    for idx, seq_ds in df.iterrows():
        bp_counter += molstru.count_ct_bptypes(seq_ds.ct, seq_ds.seq, return_tuple=False)

    total_count = float(sum(bp_counter.values()))
    for key, val in bp_counter.most_common():
        recap.update({f'{prefix}num_{key}': val})
        recap.update({f'{prefix}pct_{key}': val / total_count})

    recap.update({f'{prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))
    if plot:
        pass


def count_ct_stems(data_names,
                   recap=misc.Struct(),
                   recap_prefix='',
                   show=False,
                   warn_single=False,      # whether to warn stem length of 1.0
                   tqdm_disable=False,
                   **kwargs):
    """ count the length of stems and delta_ij in ct data and report statistics """

    df = get_midat(data_names)

    stem_counter = Counter()
    deltaij_counter = Counter()

    for idx, seq_ds in df.iterrows():
        ct = seq_ds.ct
        ct_len = len(seq_ds.ct)

        if ct_len == 0:
            logger.warning(f'No ct found for {seq_ds.file}')
        elif ct_len == 1:
            logger.warning(f'Only one ct found for {seq_ds.file}')
            print(seq_ds.ct)
            stem_counter.update([1]) # length of 1
            deltaij_counter.upate(seq_ds.ct[1] - seq_ds.ct[0])
        else:
            contig_breaks = np.nonzero(np.logical_or(
                np.diff(ct.sum(axis=1), prepend=0, append=0),   # i+j not constant
                np.diff(ct[:, 0], prepend=-2, append=-2) != 1)  # i not continuous
                )[0]
            stem_lengths = contig_breaks[1:] - contig_breaks[0:-1]

            if warn_single:
                idx_singles = np.nonzero(stem_lengths == 1)[0]
                if len(idx_singles):
                    logger.warning(f'Found stem length of 1 for file: {seq_ds.file}')
                    for _i in idx_singles:
                        print(seq_ds.ct[contig_breaks[_i]-1:contig_breaks[_i]+2])

            stem_counter.update(stem_lengths)
            deltaij_counter.update(ct[:,1] - ct[:,0])

            # stem_counter.update(np.char.add(stem_lengths.astype(str), 'S'))

    total_count = float(sum(stem_counter.values()))
    for key, val in stem_counter.most_common():
        recap.update({f'{recap_prefix}num_stem_{key}': val})
        recap.update({f'{recap_prefix}pct_stem_{key}': val / total_count})

    total_count = float(sum(deltaij_counter.values()))
    for key, val in deltaij_counter.most_common():
        recap.update({f'{recap_prefix}num_deltaij_{key}': val})
        recap.update({f'{recap_prefix}pct_deltaij_{key}': val / total_count})
    recap.update({f'{recap_prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))


def count_pseudo_knots(
        data_names,
        recap=misc.Struct(),
        prefix='',
        show=False,
        tqdm_disable=False,
        **kwargs):
    """ count the number of pseudo knots in pkldata and report statistics """

    df = get_midat(data_names)

    pknot_counter = Counter()
    # ct_contig = np.full(df['len'].max() // 2, True)

    for idx, seq_ds in df.iterrows():
        ct_len = len(seq_ds.ct)

        if ct_len < 2:
            logger.warning(f'Only {ct_len} ct found for {seq_ds.file}')
            pknot_counter.update(['0K'])
            continue

        num_pknots = molstru.count_pseudoknot(seq_ds.ct)
        if num_pknots > 10:
            logger.info(f'{num_pknots} pseudo knots found for {seq_ds.file}')

        pknot_counter.update([f'{num_pknots}K'])

    total_count = float(sum(pknot_counter.values()))
    for key, val in pknot_counter.most_common():
        recap.update({f'{prefix}num_{key}': val})
        recap.update({f'{prefix}pct_{key}': val / total_count})

    recap.update({f'{prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))


def save_individual_files(
        data_names,
        save_genre=['seq'],   # 'bpseq', 'ct', 'ctmat', 'dbn', 'ppm', 'seq' 'fasta', 'sto/stk'
        rows=None,            # the list of rows to save, default: all
        save_dir=None,        # default is the stem of the data name
        named_after='file',   # which colum for individual file names (stem is used) 
        save_header=None,     # add a header to each individual file name
        save_footer=None,     # add a footer to each individual file name
        idx_digits=None,      # number of digits for the index (None: auto)
        fasta_id='id',
        tqdm_disable=False,
        **kwargs):
    """ save the sequences into a lumpsum or individual fasta files """

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if df is None or len(df) == 0:
        logger.critical(f'Nonexisting or empty files, exiting!!!')
        return

    save_dir = Path(auto_save_prefix) if save_dir is None else Path(save_dir)

    if save_dir.is_file():
        logger.error(f'{save_dir} already exists as a file, please change!')
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    save_header = '' if save_header is None else save_header
    save_footer = '' if save_footer is None else save_footer

    if rows is None:
        rows = np.arange(df.shape[0])
    elif not hasattr(rows, '__len__'):
        rows = [rows]

    if fasta_id not in df.columns:
        logger.critical(f'dataframe does not have fasta_id col: {fasta_id} !!!')
        fasta_id = 'id'

    if named_after not in df.columns:
        logger.critical(f'dataframe does not have named_by col: {named_after} !!!')
        named_after = 'id'

    logger.info(f'Saving individual files in: {misc.str_color(save_dir)} ...')
    logger.info(f'Number of sequences to save: {len(rows)}')

    save_dir = save_dir.as_posix()
    
    save_fasta = ('fasta' in save_genre or 'fa' in save_genre) and 'seq' in df.columns

    save_seq = 'seq' in save_genre and 'seq' in df.columns
    
    save_bpseq = 'bpseq' in save_genre and ('ct' in df or 'ctmat' in df)

    
    do_ct2bpseq = 'ct' in df.columns
    do_ctmat2bpseq = 'ctmat' in df.columns

    save_stockholm = 'sto' in save_genre or 'stk' in save_genre or 'stockholm' in save_genre
    
    save_ct = 'ct' in save_genre and 'ct' in df.columns

    save_ctmat = 'ctmat' in save_genre and ('ctmat' in df or 'ct' in df)
    do_ct2ctmat = 'ctmat' not in df.columns

    save_dbn = 'dbn' in save_genre
    if save_dbn:
        # check if any 'dbn' column is an empty string
        if 'dbn' in df.columns:
            empty_dbn = df['dbn'].apply(lambda x: len(x) == 0)
            if np.any(empty_dbn):
                logger.warning(f'Found {np.sum(empty_dbn)} empty dbn, will convert from other columns')
                # df.loc[empty_dbn, 'dbn'] = df.loc[empty_dbn, 'ct'].apply(molstru.ct2dbn)
                do_ct2dbn = 'ct' in df.columns
            else:
                do_ct2dbn = False
        elif 'ct' in df.columns:
            do_ct2dbn = True
        else:
            logger.critical(f'No dbn or ct column found, cannot save dbn files!')
            save_dbn = False

    save_ppmat = 'ppmat' in save_genre and 'ppmat' in df.columns

    # if save_ct: logger.critical(f'Saving in ct format is not yet implemented!')

    fasta_id_icol = df.columns.get_loc(fasta_id)
    for i in tqdm(rows, mininterval=1, desc=save_dir, disable=tqdm_disable):
        field_name = Path(str(df.iloc[i][named_after])).stem
        save_path = os.path.join(save_dir, f'{save_header}{field_name}{save_footer}')

        if save_fasta or save_seq:
            with open(save_path + '.fasta', 'w') as iofile:
                iofile.writelines(f'>{df.iloc[i, fasta_id_icol]}\n{df.iloc[i]["seq"]}\n')

        if save_stockholm:
            sto_line = _ds2stockholm_line(i, df.iloc[i])
            with open(save_path + '.sto', 'w') as iofile:
                iofile.write(sto_line)

        if save_bpseq:
            if do_ctmat2bpseq:
                bpseq_lines = molstru.compose_ctmat2bpseq_lines(
                    df.iloc[i]['ctmat'], df.iloc[i]['seq'], id=df.iloc[i]['id'], return_list=True)
            elif do_ct2bpseq:
                bpseq_lines = molstru.compose_ct2bpseq_lines(
                    df.iloc[i]['ct'], df.iloc[i]['seq'], id=df.iloc[i]['id'], return_list=True)
            else:
                logger.critical(f'No ct and ctmat in dataframe, cannot save bpseq files!!!')

            with open(save_path + '.bpseq', 'w') as iofile:
                iofile.writelines('\n'.join(bpseq_lines))

        if save_dbn:
            if do_ct2dbn:
                dbn_str = molstru.ct2dbn(df.iloc[i]['ct'], l=df.iloc[i]['len'])
            else:
                dbn_str = df.iloc[i]['dbn']
            with open(save_path + '.dbn', 'w') as iofile:
                iofile.writelines(f'>{df.iloc[i, fasta_id_icol]}\n{df.iloc[i]["seq"]}\n{dbn_str}\n')                

        if save_ct and df.iloc[i]['ct'] is not None and len(df.iloc[i]['ct']) > 1:
            if len(df.iloc[i]['seq']) < df.iloc[i]['ct'].max():
                logger.error(f'Seq len is shorter than ct mat resnum: {df.iloc[i]["id"]}, not saved!!!')
                continue
            # ct_mat = molstru.ct2ctmat(df.iloc[i]['ct'], len(df.iloc[i]['seq']))
            # # print(f'file: {seqsdata.file[i]} appears corrupted, please check!')
            # np.savetxt(save_path + '.ct', ct_mat, fmt='%1i')

        if save_ctmat:
            if do_ct2ctmat:
                ctmat = molstru.ct2ctmat(df.iloc[i]['ct'], len(df.iloc[i]['seq']))
            else:
                ctmat = df.iloc[i]['ctmat']
            np.savetxt(save_path + '.ctmat', ctmat, fmt='%1i')

        if save_ppmat:
            np.savetxt(save_path + '.ppmat', df.iloc[i]['ppmat'], fmt='%0.8f')


def save_lumpsum_files(
        data_names,
        save_dir='./',
        save_prefix=None,
        save_csv=False,
        save_csv_index=False,
        save_csv_fmt='%8.6f',
        save_pkl=False,
        save_fasta=False,
        save_sto=False,
        save_dbn=False,
        save_unknown=False,
        save_duplicate=False,
        save_conflict=False,
        csv_exclude=['ct', 'ctmat', 'bpmat', 'ppm', 'dist'],
        pkl_exclude=None,
        fasta_id='id',        # the column for fasta id
        fasta_seq='seq',      # the column for fasta sequence
        backup=False,
        tqdm_disable=False,
        **kwargs):
    """ save all parts of midata as processed in database_* functions  """

    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    if save_dir.exists():
        if save_dir.is_dir():
            os.utime(save_dir, None)
        else:
            logger.error(f'{save_dir} already exists as a file, please change!')
            return
    else:
        save_dir.mkdir(parents=True)

        # pkl_include=['idx', 'file', 'db', 'moltype',
        #     'id', 'len', 'lenCT', 'seq', 'ct',
        #     'dataset', 'numPKnots',
        #     'resNames', 'resNameCounts', 'bpTypes', 'bpTypeCounts',
        #     'bpNums', 'bpNumCounts', 'stemLens', 'stemLenCounts',
        #     'deltaijs', 'deltaijCounts',],       # NO LONGER USED!!!!!!!!!!!!!!!
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'autosave'])

    # determine dataset fields if not yet set
    # if 'save2lib' in df.columns:
    #     idx_save2lib = df.index[df.save2lib]
    # else:
    #     idx_save2lib = df.index

    # if split_data:
    #     dataset_names = split_names,
    #     idx_set0, idx_set1 = train_test_split(idx_save2lib, train_size=split_size, shuffle=False,
    #                         random_state=20151029)
    #     df.loc[idx_set0, 'dataset'] = dataset_names[0]
    #     df.loc[idx_set1, 'dataset'] = dataset_names[1]
    # else:
    #     dataset_names = [save_name]
    #     df.loc[idx_save2lib, 'dataset'] = dataset_names[0]

    # save csv
    if save_csv:
        csv_columns = list(filter(lambda _s: _s not in csv_exclude, df.columns.to_list()))
        csv_kwargs = dict(index=save_csv_index, float_format=save_csv_fmt, columns=csv_columns)
        csv_file = save_dir / (save_prefix + '.csv')
        logger.info(f'Storing csv: {misc.str_color(csv_file)} of shape: {df.shape}')
        if backup:
            csv_file = gwio.new_path_with_backup(csv_file)
        df.to_csv(csv_file, **csv_kwargs)

    # save sequences with unknown residues
    if save_unknown:
        if 'unknownSeq' not in df.columns:
            logger.critical(f'save_unknown requires column: unknownSeq, but not found!!!')
        else:
            idx_unknown_seqs = df.index[df.unknownSeq]
            num_unknown_seqs = len(idx_unknown_seqs)
            logger.info(f'Number of sequences with unknown residues: {num_unknown_seqs}')
            if num_unknown_seqs > 0:
                unk_file = save_dir / (save_prefix + '_unknown')
                df_unk = df.loc[idx_unknown_seqs]
                logger.info(f'Storing unknown sequences as: {misc.str_color(unk_file)} of shape: {df_unk.shape}')
                if save_csv: df_unk.to_csv(unk_file.with_suffix('.csv'), **csv_kwargs)
                if save_pkl: df_unk.to_pickle(unk_file.with_suffix('.pkl'))
                # if save_lib: save_lib_files(df, save_dir=dir_tmp)

    # save conflict
    if save_conflict:
        if 'conflictVal' not in df.columns and 'conflictSeq' not in df.columns:
            logger.critical(f'save_conflict requires column: conflictSeq or conflictVal, but not found!!!')
        else:
            idx_conflicts = None
            if 'conflictVal' in df.columns:
                idx_conflicts = df.conflictVal
            if 'conflictSeq' in df.columns:
                idx_conflicts = df.conflictSeq if idx_conflicts is None else (idx_conflicts | df.conflictSeq)

            idx_conflicts = df.index[idx_conflicts]
            num_conflict_seqs = len(idx_conflicts)
            logger.info(f'Number of conflict sequences: {num_conflict_seqs}')
            if num_conflict_seqs > 0:
                unk_file = save_dir / (save_prefix + '_conflict')
                df_tmp = df.loc[idx_conflicts]
                logger.info(f'Storing conflict sequences as: {misc.str_color(unk_file)} of shape: {df_tmp.shape}')
                if save_csv: df_tmp.to_csv(unk_file.with_suffix('.csv'), **csv_kwargs)
                if save_pkl: df_tmp.to_pickle(unk_file.with_suffix('.pkl'))
                # if save_lib: save_lib_files(df, save_dir=dir_tmp)

    # save duplicate
    if save_duplicate:
        if 'duplicateSeq' not in df.columns:
            logger.critical(f'save_duplicate requires column: duplicateSeq, but not found!!!')
        else:
            idx_duplicate_seqs = df.index[df.duplicateSeq]
            num_duplicate_seqs = len(idx_duplicate_seqs)
            logger.info(f'Number of duplicate sequences: {num_duplicate_seqs}')
            if num_duplicate_seqs > 0:
                unk_file = save_dir / (save_prefix + '_duplicate')
                df_tmp = df.loc[idx_duplicate_seqs]
                logger.info(f'Storing duplicate sequences as: {misc.str_color(unk_file)} of shape: {df_tmp.shape}')
                if save_csv: df_tmp.to_csv(unk_file.with_suffix('.csv'), **csv_kwargs)
                if save_pkl: df_tmp.to_pickle(unk_file.with_suffix('.pkl'))
                # if save_lib: save_lib_files(df, save_dir=dir_tmp)

    # # drop rows & columns
    # if 'save2lib' in df.columns:
    #     df.drop(index=df.index[~df.save2lib], inplace=True)

    # df.drop(columns=list(set(df.columns.to_list()) - set(pkl_include)), inplace=True)
    # recap.num_seqs_saved = df.shape[0]

    if save_pkl:
        pkl_file = save_dir / (save_prefix + '.pkl')
        logger.info(f'Storing pickle: {misc.str_color(pkl_file)} of shape: {df.shape} ...')
        if backup:
            pkl_file = gwio.new_path_with_backup(pkl_file)
        if pkl_exclude:
            pkl_columns = list(filter(lambda _s: _s not in pkl_exclude, df.columns.to_list()))
            pd.to_pickle(df[pkl_columns], pkl_file)
        else:
            df.to_pickle(pkl_file)
        # gwio.pickle_squeeze(df_tmp.to_dict(orient='list'), pkl_file, fmt='lzma')

    if save_fasta:
        fasta_file = save_dir / (save_prefix + '.fasta')
        logger.info(f'Storing fasta file: {misc.str_color(fasta_file)} of counts: {len(df)} ...')
        # fasta_lines = itertools.starmap(lambda idx, ds: f'>{ds.idx}_{ds.id}\n{ds.seq}\n',
        #     tqdm(df.iterrows(), total=len(df), desc='Get fasta lines'))
        # fasta_lines = misc.mpi_map(_ds2fasta_line,
        #     tqdm(df.iterrows(), total=len(df), desc='Get fasta lines', disable=tqdm_disable),
        #     quiet=True, starmap=True)

        fasta_lines = misc.mpi_map(functools.partial(_ds2fasta_line, id_key=fasta_id, seq_key=fasta_seq), 
            df.iterrows(), total=len(df), desc='Get fasta lines', starmap=True, tqdm_disable=False)
        if backup:
            fasta_file = gwio.new_path_with_backup(fasta_file)

        with open(fasta_file, 'w') as iofile:
            # for _i in tqdm(range(len(df)), desc='Saving Fasta'):
                # fasta_lines = f'>{df.iloc[_i]["idx"]}_{df.iloc[_i]["id"]}\n{df.iloc[_i]["seq"]}\n'
            iofile.writelines(fasta_lines)

    if save_dbn:
        dbn_file = save_dir / (save_prefix + '.dbn')
        logger.info(f'Storing dbn file: {misc.str_color(dbn_file)} of counts: {len(df)} ...')

        dbn_lines = misc.mpi_map(functools.partial(_ds2dbn_line, id_key=fasta_id, seq_key=fasta_seq),
            df.iterrows(), total=len(df), desc='Get dbn lines', starmap=True, tqdm_disable=False)

        if backup:
            dbn_file = gwio.new_path_with_backup(dbn_file)

        with open(dbn_file, 'w') as iofile:
            iofile.writelines(dbn_lines)

    if save_sto:
        sto_file = save_dir / (save_prefix + '.sto')
        logger.info(f'Storing stockholm file: {misc.str_color(sto_file)} of counts: {len(df)} ...')

        sto_lines = misc.mpi_map(_ds2stockholm_line, df.iterrows(), total=len(df), 
            desc='Get sto lines', starmap=True, tqdm_disable=False)

        if backup:
            sto_file = gwio.new_path_with_backup(sto_file)
            
        with open(sto_file, 'w') as iofile:
            iofile.writelines(sto_lines)        
        

def _ds2stockholm_line(index, ds):
    if 'dbn' in ds and len(ds.dbn) == len(ds.seq):
        wussn = ds.dbn
    elif 'ct' in ds and len(ds.ct):
        wussn = molstru.ct2dbn(ds.ct, len(ds.seq))
        assert len(ds.seq) == len(wussn), 'seq and wussn must be of the same length!!!'
    else:
        wussn = None        
    return '\n'.join(molstru.compose_stockholm_lines(ds.seq, id=ds['id'], wussn=wussn)) + '\n'

def _ds2fasta_line(index, ds, id_key='id', seq_key='seq'):
    """ only for the mpi_map in save_lumpsum_files """
    # if 'idx' in ds:
    #     return f'>{ds.id.strip()}\n{ds.seq}\n'
    # else:
    #     return f'>{ds.id.strip()}\n{ds.seq}\n'
    return f'>{ds[id_key]}\n{ds[seq_key]}\n'

def _ds2dbn_line(index, ds, id_key='id', seq_key='seq'):
    """ only for the mpi_map in save_lumpsum_files """
    # if 'idx' in ds:
    #     return f'>{ds.id.strip()}\n{ds.seq}\n'
    # else:
    #     return f'>{ds.id.strip()}\n{ds.seq}\n'
    if 'dbn' in ds and len(ds.dbn) == len(ds[seq_key]):
        return f'>{ds[id_key]}\n{ds[seq_key]}\n{ds.dbn}\n'
    elif 'ct' in ds and ds.ct is not None:
        dbn = molstru.ct2dbn(ds.ct, len(ds[seq_key]))
        assert len(ds[seq_key]) == len(dbn), 'seq and dbn must be of the same length!!!'
        return f'>{ds[id_key]}\n{ds[seq_key]}\n{dbn}\n'
    else:
        logger.error(f'Cannot generate dbn for seq: {ds.seq} !!!')
        return None


def save_files(data_names, 
        args=misc.Struct(),
        save_dir='./',
        save_prefix=None,            # default as data_names[0].stem
        save_lumpsum=True,           # whether to save lumpsum files
        save_lib=False,              # save entries with save2lib=True
        save_individual=False,       # whether to save individual files
        save_individual_dir=None,    # default to save_prefix
        save_individual_prefix=None, # defautt to None
        save_args=True,
        **kwargs):
    """ a portal for saving midat in both lumpsum and individual forms """

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)

    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'autosave'])
    if type(save_prefix) in (list, tuple):
        logger.warning(f'save_prefix is not a string: {save_prefix}')
        save_prefix = save_prefix[0]
        
    if save_lumpsum:
        save_lumpsum_files(df, recap=args, save_dir=save_dir, save_prefix=save_prefix, **kwargs)

    if save_lib:
        if "save2lib" in df.columns:
            df_lib = df[df['save2lib'] == True]
        elif 'seq' in df.columns:
            logger.warning(f'save2lib not in df.columns, use unique sequences!')
            df_lib = df.groupby(by='seq').head(1)
        else:
            logger.warning(f'Neither save2lib or seq exists in df.columns, use all!!')
            df_lib = df

        args.num_lib_seqs = len(df_lib)
        count_all_residues(df_lib, args=args, prefix='lib_', show=False)
        save_lumpsum_files(df_lib, recap=args,
            save_dir=save_dir, save_prefix="libset" if save_prefix == 'allset' else (save_prefix + '_lib'), **kwargs)

    if save_individual:
        save_individual_files(df, save_header=save_individual_prefix, **kwargs,
            save_dir=save_prefix if save_individual_dir is None else save_individual_dir)

    # save json
    if save_args and len(args):
        args_file = save_prefix + '.args'
        if save_dir is not None: args_file = Path(save_dir) / args_file
        logger.info(f'Storing args as json: {args_file}')
        gwio.dict2json(vars(args), args_file)


def check_unknown_residues(
            data_names,
            recap=misc.Struct(),
            save2lib_only=True,
            **kwargs):
    """ check and tag entries for unknown residue names """
    # NOTE: all rows are kept for now. If too slow, one can drop invalid rows:
    # df_all.drop(index=df_all.index[~ df_all.save2lib], inplace=True)
    # recap_json.num_consistent_seq_ct = df_all.shape[0]
    # logger.info(f'{df_all.shape[0]} items with valid seq and ct out of ' + \
    #             f'{recap_json.num_seqs} total')

    # keys are the identities checked for duplication
    # vals are the columns whose values are further checked for value duplication

    df = get_midat(data_names)

    logger.info('Checking for unknown residue letters...')
    known_resnames = set(reslet_lookup.keys())
    recap.num_unknown_seqs = 0
    recap.known_resnames = ''.join(list(reslet_lookup.keys()))

    if 'unknownSeq' not in df.columns: df['unknownSeq'] = False
    if 'save2lib' not in df.columns: df['save2lib'] = True

    for i, seq in enumerate(df.seq):
        if len(set(seq) - known_resnames):
            recap.num_unknown_seqs += 1
            df.loc[i, 'unknownSeq'] = True
            df.loc[i, 'save2lib'] = False
    logger.info(f'Found {recap.num_unknown_seqs} sequences with unknown residues')

    return df, recap


def check_duplicate_keyvals(
            data_names,
            recap=misc.Struct(),
            save2lib_only=True,
            keys='seq', vals=None,
            **kwargs):
    """ check and tag entries for duplication, first by keys then by vals
        Note: the first entry with duplicate keys is now saved regardless
              of whether their vals are conflicting
    """
    # NOTE: all rows are kept for now. If too slow, one can drop invalid rows:
    # df_all.drop(index=df_all.index[~ df_all.save2lib], inplace=True)
    # recap_json.num_consistent_seq_ct = df_all.shape[0]
    # logger.info(f'{df_all.shape[0]} items with valid seq and ct out of ' + \
    #             f'{recap_json.num_seqs} total')

    # keys are the identities checked for duplication
    # vals are the columns whose values are further checked for value duplication

    df = get_midat(data_names)

    if vals is None:
        vals = []
    elif isinstance(vals, str):
        vals = [vals]

    logger.info('Checking for sequence/ct duplicates...')
    recap.keys = keys
    recap.vals = vals
    recap.num_conflict_grps = 0
    recap.num_conflict_vals = 0
    recap.num_duplicate_grps = 0
    recap.num_duplicate_seqs = 0
    recap.num_duplicate_seqvals = 0
    recap.num_duplicate_seqs_removed = 0

    if 'duplicateSeq' not in df: df['duplicateSeq'] = False
    if 'idxSameSeq' not in df: df['idxSameSeq'] = None
    if 'idxSameSeqVal' not in df: df['idxSameSeqVal'] = None
    if 'conflictVal' not in df : df['conflictVal'] = False
    if 'save2lib' not in df: df['save2lib'] = True

    if save2lib_only:
        df_grps = df[df.save2lib].groupby(by=keys)
    else:
        df_grps = df.groupby(by=keys)

    for seq, df_one_grp in tqdm(df_grps, desc='Checking duplicates'):
        if len(df_one_grp) == 1: continue

        num_seqs = len(df_one_grp)
        recap.num_duplicate_grps += 1
        recap.num_duplicate_seqs += num_seqs

        df.loc[df_one_grp.index, 'duplicateSeq'] = True
        df.loc[df_one_grp.index, 'idxSameSeq'] = df_one_grp.iloc[0]['idx']
        # df.loc[df_one_grp.index[1:], 'save2lib'] = False # keep 1st only

        all_same_vals = True
        for val in vals:
            val_1st = df_one_grp.iloc[0][val]
            same_grp_vals = [True] * num_seqs
            for igrp in range(1, len(df_one_grp)):
                same_grp_vals[igrp] = np.array_equal(val_1st, df_one_grp.iloc[igrp][val])

            if same_grp_vals.count(True) > 1:
                df.loc[df_one_grp.index[same_grp_vals], 'idxSameSeqVal'] = df_one_grp.iloc[0]['idx']

            if not all(same_grp_vals): # conflicting values within the same group
                all_same_vals = False
                recap.num_conflict_grps += 1
                recap.num_conflict_vals += num_seqs
                df.loc[df_one_grp.index, 'conflictVal'] = True
                # df.loc[df_one_grp.index, 'save2lib'] = False
                logger.warning(f'The same {keys} but different {vals} for the following:')
                print(seq)
                print(df_one_grp[['idx', 'file']])

        # keep the 1st value now, whether with the same or different values
        if all_same_vals:
            df.loc[df_one_grp.index[1:], 'save2lib'] = False
        else:
            df.loc[df_one_grp.index[1:], 'save2lib'] = False

    recap.num_duplicate_seqvals += (df.idxSameSeqVal > 0).to_numpy().astype(int).sum()
    recap.num_duplicate_seqs_removed = (df.duplicateSeq & (~ df.save2lib)).to_numpy().astype(int).sum()

    logger.info(f'Found {recap.num_duplicate_seqs} duplicate sequences in {recap.num_duplicate_grps} groups')
    logger.info(f'Found {recap.num_conflict_vals} conflicting values in {recap.num_conflict_grps} groups')

    return df, recap


def chew_midat(
        data_names,
        min_len=None,
        max_len=None,
        fname2idx=None,              # convert fname column to indices (usually filename)
        seq2rna=False,               # convert to RNA sequences
        seq2upper=False,             # conver to upper case
        select_row=None,             # select passed rows
        get_duplicate=None,          # get duplicated rows by the passed column
        get_unique=None,             # select rows with unique values by the passed column
        include_lst=None,            # include rows with column values passed via a file, [column, lst_file(s)]
        exclude_lst=None,            # exclude rows with column values passed via a file, [column, lst_file(s)]               
        include_val=None,            # include rows with column values passed via cmdline, [column, value(s)]
        exclude_val=None,            # exclude rows with column values passed via cmdline, [column, value(s)]
        include_seq=None,            # include rows with sequences in the passed fasta file or seq_str(s)
        exclude_seq=None,            # exclude rows with sequences in the passed fasta file or seq_str(s)
        range_col=None,              # filter rows with column values in a numeric range, [column, min, [max]]
        remove_noncanon=False,       # remove non-canonical base pairs in RNA secondary structures
        remove_pknot=False,          # remove pseudoknots in RNA secondary structures
        split_groupby=None,          # split the dataframe by the groupby col, giving -[groupby]-IS-[val]
        split_groupby_reverse=False, # reverse the groupby selection, giving -[groupby]-SUB-[val]
        tqdm_disable=False,          # NOTE: see save_all_files for saving args!!!
        **kwargs):
    """ comb through midata for various tasks such as selection and split """

    args = misc.Struct(locals()) # it will be saved as recap
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    auto_save_prefix = misc.get_1st_value([
        kwargs.get('save_prefix', None),
        auto_save_prefix,
        'refine_midat'])

    df_has_changed = False
    args.df_in_shape = df.shape

    if fname2idx is not None and fname2idx in df.columns:
        logger.info(f'Converting file names to indices: {fname2idx}...')
        df['idx'] = df[fname2idx].apply(lambda x: Path(x).stem.split('_')[0]).astype(int)
        df_has_changed = True
    elif fname2idx is not None:
        logger.warning(f'fname2idx: {fname2idx} not found in df.columns, ignored!')

    if seq2upper:
        logger.info('Converting all sequences to upper case...')
        df['seq'] = df['seq'].str.upper()
        df_has_changed = True

    if seq2rna:
        df['seq'] = misc.mpi_map(molstru.seq2RNA, df['seq'], tqdm_disable=tqdm_disable)
        df_has_changed = True

    # min and max len
    num_rows = df.shape[0]
    if min_len is not None or max_len is not None:
        df_len_range = [df['len'].min(), df['len'].max()]
        logger.info(f'Length range: {df_len_range}')
        if min_len is None:
            min_len = df_len_range[0]
        else:
            df = df[df['len'] >= min_len]
            logger.info(f'Applied len>={min_len}, after shape: {df.shape}')
        if max_len is None:
            max_len = df_len_range[1]
        else:
            df = df[df['len'] <= max_len]
            logger.info(f'Applied len<={max_len}, after shape: {df.shape}')

        auto_save_prefix += f'_len{min_len}-{max_len}'
        if num_rows != df.shape[0]: 
            df_has_changed = True

    # column ranges
    if range_col is not None:
        logger.info(f'Applying column ranges: {range_col}...')
        auto_save_prefix += f'_{range_col[0]}-{range_col[1]}'
        df = df[df[range_col[0]] >= float(range_col[1])]
        logger.info(f'{df.shape[0]} rows with {range_col[0]}>={range_col[1]}')
        if len(range_col) > 2:
            auto_save_prefix += f'-{range_col[2]}'
            df = df[df[range_col[0]] <= float(range_col[2])]
            logger.info(f'{df.shape[0]} rows with {range_col[0]}<={range_col[2]}')            
            
    # unique rows
    num_rows = df.shape[0]
    if get_unique is not None:
        auto_save_prefix += '_unique'
        df = df.groupby(by=get_unique).head(1)
        if num_rows != df.shape[0]: 
            df_has_changed = True
        logger.info(f'Applied unique by [{get_unique}], shape: {df.shape}')

    # duplicate rows
    num_rows = df.shape[0]
    if get_duplicate is not None:
        df_grp_all = df.groupby(by=get_duplicate) # .filter(lambda x: len(x) > 1)
        auto_save_prefix += '_duplicate'
        for key, df_grp in df_grp_all:
            if len(df_grp) == 1: continue

            val = df_grp.iloc[0]['ct']
            for i in range(1, len(df_grp)):
                if not np.array_equal(val, df_grp.iloc[i]['ct']):
                    print(df_grp['id'])
        # df_dup = df[df.duplicated(subset=duplicate, keep=False)]
        if num_rows != df.shape[0]: 
            df_has_changed = True

    # remove non-canonical base pairs
    if remove_noncanon:
        df['ct'] = misc.mpi_map(molstru.remove_noncanonical_in_ct, list(zip(df['ct'], df['seq'])),
            starmap=True, tqdm_disable=tqdm_disable, desc='Remove noncanonicals in ct')
        
    # remove pseudoknots
    if remove_pknot:
        df['dbn'] = misc.mpi_map(molstru.remove_pseudoknot_by_nkn, list(zip(df['dbn'], df['nkn'])),
            starmap=True, tqdm_disable=tqdm_disable, desc='Remove pseudoknot in DBN')
        df['ct'] = misc.mpi_map(molstru.dbn2ct, df['dbn'], tqdm_disable=tqdm_disable, desc='DBN2CT')

    # select_row via idx
    num_rows = df.shape[0]
    if select_row is not None:
        auto_save_prefix += f'_rowselect-{len(select_row)}'
        df = df.take(select_row, axis=0)
        if num_rows != df.shape[0]: 
            df_has_changed = True

    # include column in lst_file
    num_rows = df.shape[0]
    if include_lst is not None and len(include_lst) > 1:
        logger.info(f'Applying include_lst: {include_lst}...')
        # col_values = gwio.get_file_lines(include_lst[1], strip=True, keep_empty=False, keep_comment=False)
        col_values = []
        for _val in include_lst[1:]:
            if os.path.isfile(_val):
                col_values.extend(gwio.get_file_lines(_val, strip=True, keep_empty=False, keep_comment=False))
            else:
                col_values.append(_val)

        if len(col_values):
            logger.info(f'Passed lists: {include_lst[1:]} has {len(col_values)} lines...')
            auto_save_prefix += f'_{include_lst[0]}-IS-{col_values[0]}' if len(col_values) == 1 else \
                                f'_{include_lst[0]}-IN-{"-".join([col_values[0], col_values[-1]])}'

            df = df[df[include_lst[0]].str.strip().isin(col_values)]

            logger.info(f'{df.shape[0]} rows left after include_lst: {include_lst}')
            if num_rows != df.shape[0]: 
                df_has_changed = True
        else:
            logger.error(f'No values found in passed vals: {include_lst[1:]}!!!')
    elif include_lst is not None:
        logger.error(f'Only column name is passed for include_lst: {include_lst}!!!')
            
    # exclude column in lst_file
    num_rows = df.shape[0]
    if exclude_lst is not None and len(exclude_lst) > 1:
        logger.info(f'Applying exclude_lst: {exclude_lst}...')
        col_values = []
        for _val in exclude_lst[1:]:
            if os.path.isfile(_val):
                col_values.extend(gwio.get_file_lines(_val, strip=True, keep_empty=False, keep_comment=False))
            else:
                col_values.append(_val)
                
        if len(col_values):
            logger.info(f'Passed lists: {exclude_lst[1:]} has {len(col_values)} lines...')
            auto_save_prefix += f'_{exclude_lst[0]}-SUB-{col_values[0]}' if len(col_values) == 1 else \
                                f'_{exclude_lst[0]}-OUT-{"-".join([col_values[0], col_values[-1]])}'
                                
            df = df[~df[exclude_lst[0]].str.strip().isin(col_values)]

            logger.info(f'{df.shape[0]} rows left after exclude_lst: {exclude_lst}')
            if num_rows != df.shape[0]: 
                df_has_changed = True
        else:
            logger.error(f'No values found in passed vals: {exclude_lst[1:]}!!!')
    elif exclude_lst is not None:
        logger.error(f'Only column name is passed for include_lst: {exclude_lst}!!!')

    # select by seq
    num_rows = df.shape[0]
    if include_seq is not None:
        auto_save_prefix += '_seqinclude'
        if isinstance(include_seq, str) and Path(include_seq).exists():
            seq_data = molstru.SeqStruct2D(include_seq, fmt='fasta')
            include_seq = seq_data.seq
        if isinstance(include_seq, str):
            include_seq = [include_seq]

        df = df[df['seq'].isin(include_seq)]
        if num_rows != df.shape[0]: 
            df_has_changed = True
        logger.info(f'Applied isin(sequence), after shape: {df.shape}')

    # select by seq
    num_rows = df.shape[0]
    if exclude_seq is not None:
        auto_save_prefix += '_seqexclude'
        if isinstance(exclude_seq, str) and Path(exclude_seq).exists():
            seq_data = molstru.SeqStruct2D(exclude_seq, fmt='fasta')
            exclude_seq = seq_data.seq
        if isinstance(exclude_seq, str):
            exclude_seq = [exclude_seq]

        df = df[~df['seq'].isin(exclude_seq)]
        if num_rows != df.shape[0]:
            df_has_changed = True
        logger.info(f'Applied ~isin(sequence), after shape: {df.shape}')        

    # include column values
    num_rows = df.shape[0]
    if include_val is not None and len(include_val) > 1:
        auto_save_prefix += f'_{include_val[0]}-IS-{"_".join(include_val[1:])}'
        logger.info(f'Applying include_val: {include_val}...')
        df = df[df[include_val[0]].isin(include_val[1:])]
        logger.info(f'{df.shape[0]} rows left after select_include')
        if num_rows != df.shape[0]: 
            df_has_changed = True

    # exclude column values
    num_rows = df.shape[0]
    if exclude_val is not None and len(exclude_val) > 1:
        auto_save_prefix += f'_{exclude_val[0]}-SUB-{"_".join(exclude_val[1:])}'
        logger.info(f'Applying exclude_val: {exclude_val}...')
        df = df[~df[exclude_val[0]].isin(exclude_val[1:])]
        logger.info(f'{df.shape[0]} rows left after select_exclude')
        if num_rows != df.shape[0]: 
            df_has_changed = True

    args.df_out_shape = df.shape
    kwargs.setdefault('save_prefix', auto_save_prefix)

    # save the main df
    if split_groupby is None:
        kwargs.setdefault('save_pkl', df_has_changed)
        save_files(df, args=args, tqdm_disable=tqdm_disable, **kwargs)
    else: # split_groupby and save each group
        kwargs.setdefault('save_pkl', True)
        if split_groupby not in df.columns:
            logger.critical(f'split_groupby: {split_groupby} not in df.columns!!!')
        else:
            df_grps = df.groupby(by=split_groupby)
            save_prefix = kwargs.pop('save_prefix')
            kwargs['save_json'] = False
            for key, df_one_grp in df_grps:
                logger.info(f'Saving split_groupby, key: {split_groupby}, val: {key}, count: {len(df_one_grp)} ...')
                if split_groupby_reverse:
                    save_files(df.drop(index=df_grps.groups[key], inplace=False), 
                        save_prefix=f'{save_prefix}_{split_groupby}-SUB-{key}',
                        tqdm_disable=tqdm_disable, **kwargs)
                else:
                    save_files(df_one_grp, save_prefix=f'{save_prefix}_{split_groupby}-IS-{key}',
                        tqdm_disable=tqdm_disable, **kwargs)

    return df


def split_seqs(data_names,
               seq_fmt='fasta',
               num_chunks=10,
               num_seqs=None,
               **kwargs):
    """ split sequences into multiple files (much slower than slurp_1p.sh fasta_split) """

    if isinstance(data_names, str): data_names = [data_names]

    save_prefix = Path(data_names[0]).stem

    seqs_obj = molstru.SeqStruct2D()
    for data_name in data_names:
        seqs_obj.parse_files(data_name, fmt=seq_fmt)

    if num_seqs is None:
        num_seqs = math.ceil(len(seqs_obj.id) / num_chunks)

    for ifile in range(0, len(seqs_obj.id), num_seqs):
        save_path = f'{save_prefix}_{ifile+1}.fasta'
        logger.info(f'Saving fasta file: {save_path}...')
        with open(save_path, 'w') as iofile:
            iofile.writelines(
                seqs_obj.get_fasta_lines(list(range(ifile, ifile + num_seqs)), line_break=True)
                )


def split_midat_tvt(data_names,
                   fractions=[0.70, 0.15], # train and valid fractions, test fraction = 1 - sum(fractions)
                   save_prefixes=None,     # default: [test, train-valid, valid, train]. Four required if provided!
                   tqdm_disable=False,     # NOTE: see save_all_files for saving args!!!
                   **kwargs):
    """ train-valid-test split via iterative split_midat(), take stratify, bucket_key, bucket_num ... """

    recap = misc.Struct(locals())
    logger.debug('Arguments:\n' + gwio.json_str(recap.__dict__))

    assert len(fractions) > 1, f"At least two fractions: {fractions} must be provided!"
    assert fractions[0] + fractions[1] < 1., f"The sum of the first two fractions must < 1!"

    midat, auto_save_prefix = get_midat(data_names, return_save_prefix=True)

    if save_prefixes is None:
        save_prefix = misc.get_1st_value([auto_save_prefix, inspect.currentframe().f_code.co_name])
        save_prefixes = [f'{save_prefix}_{_s}' for _s in ['test', 'train-valid', 'valid', 'train']]

    tvt_list = []

    train_valid_set, test_set = split_midat(midat, fraction=fractions[0]+fractions[1],
        tqdm_disable=tqdm_disable, save_prefixes=False, **kwargs)
    tvt_list.append(test_set)
    tvt_list.append(train_valid_set)

    train_set, valid_set = split_midat(train_valid_set, fraction=fractions[0]/(fractions[0]+fractions[1]),
        tqdm_disable=tqdm_disable, save_prefixes=False, **kwargs)
    tvt_list.append(valid_set)
    tvt_list.append(train_set)

    if save_prefixes is not False and len(save_prefixes) == len(tvt_list):
        for i in range(len(tvt_list)):
            save_files(tvt_list[i], save_prefix=save_prefixes[i],
                tqdm_disable=tqdm_disable, **kwargs)
    else:
        logger.info(f'No saving with save_prefixes: {save_prefixes}.')

    return tvt_list


def split_midat_cv(data_names,
                   nfold=5,            # use split_midat(), take stratify, bucket_key, bucket_num
                   save_prefixes=None, # _cv{i} suffix will be added
                   tqdm_disable=False, # NOTE: see save_all_files for saving args!!!
                   **kwargs):
    """ cross-validation split via iterative split_midat(), take stratify, bucket_key, bucket_num ..."""

    recap = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    train_set, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if save_prefixes is None:
        save_prefix = misc.get_1st_value([auto_save_prefix, 'split_midat_cv'])
        save_prefixes = [f'{save_prefix}_cv{i}' for i in range(1, nfold + 1)]

    midat_list = []
    for i in range(nfold - 1):
        out_fraction = 1. / nfold / (1. - i / nfold)
        train_set, test_set = split_midat(train_set, fraction=1 - out_fraction,
            tqdm_disable=tqdm_disable, save_prefixes=False, **kwargs)
        midat_list.append(test_set)
    midat_list.append(train_set)

    if save_prefixes is not False and len(save_prefixes) == nfold:
        for i in range(nfold):
            save_files(midat_list[i], save_prefix=save_prefixes[i],
                tqdm_disable=tqdm_disable, **kwargs)
    else:
        logger.info(f'No saving with save_prefixes: {save_prefixes}.')

    return midat_list


def split_midat(data_names,
                fraction=0.85,          # 1st fraction (e.g., 85/15 for train+valid/test, 0.8235/0.1765 for train/valid)
                stratify=None,          # the stratify column
                bucket_key=None,        # divide the key col into buckets
                bucket_num=11,          # number of buckets
                shuffle=False,          # only used in train_test_split
                random_state=None,
                save_prefixes=None,     # two names needed
                tqdm_disable=False,     # NOTE: see save_all_files for saving args!!!
                **kwargs):
    """ TWO-WAY-ONLY split based on stratify and/or bucket_key/num """

    recap = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if save_prefixes is None:
        save_prefix = misc.get_1st_value([auto_save_prefix, 'split_midat'])
        save_prefixes = [f'{save_prefix}_train', f'{save_prefix}_test']

    df_list = [] # to be returned
    stratify_list = [None]  # this is passed to train_test_split() via stratify=stratify_list[-1]
    bucketize = bucket_key is not None and bucket_num > 1

    if stratify and stratify not in df.columns:
        if stratify == 'none':
            logger.warning(f'"none" was passed as stratify column, will not use stratify...')
            stratify = None
        else:
            logger.critical(f'stratify: {stratify} not in df.columns!!!')
            return df_list

    if stratify:
        logger.info(f'Using column:{stratify} for stratified train_test_split...')
        shuffle = True
        stratify_list.append(df[stratify])

    if bucketize:
        shuffle=True
        df['splitBucket'] = 0
        # the index can be "corrupted" (e.g., duplicates) by dataframe concatation, etc.
        df.reset_index(inplace=True, drop=True)
        if stratify:
            df_grps = df.groupby(by=stratify)
        else:
            df_grps = [('all', df)] # mimic pd.grouby return

        for key, df_one_grp in df_grps:
            grp_size = len(df_one_grp)
            nbins = min([bucket_num, int(grp_size * min([fraction, 1 - fraction]) / 2)])
            if stratify:
                logger.info(f'Divide {bucket_key} into {nbins} buckets for {stratify}={key} with {grp_size} samples ...')
            else:
                logger.info(f'Divide {bucket_key} into {nbins} buckets for a total of {grp_size} samples ...')

            if nbins < 2:
                logger.debug(f'Only one bucket for {key} with {grp_size} samples!')
                # df.loc[df_one_grp.index, 'splitBucket'] = 0
            else:
                bucket_intervals, bucket_grids = pd.qcut(df_one_grp[bucket_key], nbins, retbins=True, duplicates='drop', precision=1)
                logger.info(f'Bucket grids: {np.array2string(bucket_grids, precision=1, floatmode="fixed")}')
                # convert bucket_interval from a category dictionary to index
                bucket_idx = bucket_intervals.cat.codes
                df.loc[df_one_grp.index, 'splitBucket'] = bucket_idx.to_list()
                # df.loc[df_one_grp['__init_index__'].values]['splitBucket'] = bucket_idx.to_list()

        if stratify:
            new_col = f'{stratify}+splitBucket'
            df[new_col] = df[stratify].astype(str) + '+' + df['splitBucket'].astype(str)
            stratify_list.append(df[new_col])
        else:
            stratify_list.append(df['splitBucket'].astype(str))

        # df.drop(columns=['__init_index__'], inplace=True)
    while len(stratify_list):
        try:
            df_list = train_test_split(df,
                train_size=fraction,
                stratify=stratify_list.pop(-1),
                shuffle=shuffle,
                random_state=random_state)
            break
        except:
            logger.critical(f'Failed to train_test_split with last stratify, try coarser...')

    # data are now stored in df_list and save_name_list
    if save_prefixes and len(df_list):
        kwargs.setdefault('save_pkl', True)
        for i, save_name in enumerate(save_prefixes):
            save_files(df_list[i], args=recap, save_prefix=save_name,
                tqdm_disable=tqdm_disable, **kwargs)

    return df_list


def manifold_pairwise_mat(
        inputs=Path.home() / 'database/contarna/strive_2022/nr80-vs-nr80.rnadistance.alnIdentity_pairwise.pkl',
        label_input=Path.home() / 'database/contarna/strive_2022/libset_len30-600.pkl',
        label_key='moltype',     # the key col to extract from the meta_input
        square_dataframe=False,  # whether to square up the rows and columns of the dataframe
        diagonal=1.0,            # fill diagonal of the matrix before clustering
        diagonal_nan=None,       # fill diagonal NaNs with this value
        conjugate_to_nan=True,   # fill asymmetric NaNs with their conjugate values
        symmetric_nan=None,      # fill symmetric NaNs with this value
        nan=None,                # fill all NaNs with this value
        sim2dist=True,           # convert similarity to distance
        algorithm='umap',        # 'affinity' or 'optics' or 'dbscan' or 'hdbscan'
        transform_nans=True,     # transform ids with NaNs into embedding space
        save_dir=None,           # save to the current dir if not specified
        save_prefix=None,        # using the input file name if not specified
        save_pkl=False,
        show_fig=False,
        show_img=True,
        **kwargs):
    """ cluster pariwise matrix which is assumed to be symmetric and square """
    import umap

    args = misc.Struct(locals())
    # logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(inputs, return_save_prefix=True)
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    save_prefix = misc.get_1st_value([
        save_prefix,
        f'{auto_save_prefix}' if auto_save_prefix else None,
        inspect.currentframe().f_code.co_name])
    
    # extract numpy array from dataframe
    pairwise_idx = np.arange(len(df), dtype=int)
    logger.info(f'Calling bake_pairwise_df() with diagonal_nan: {diagonal_nan}, diagonal: {diagonal}, conjugate_to_nan: {conjugate_to_nan}, symmetric_nan: {symmetric_nan} ...')
    pairwise_mat, pairwise_ids, irows_with_NaNs = bake_pairwise_df(df,
                square_dataframe=square_dataframe,
                diagonal=diagonal, 
                diagonal_nan=diagonal_nan,
                nan=nan,
                conjugate_to_nan=conjugate_to_nan, 
                symmetric_nan=symmetric_nan,
                symmetrize=True,
                save_mat=False, 
                save_pkl=False)

    # Find the labels for each data point
    pairwise_labels = None
    if label_input is not None and Path(label_input).exists():
        label_df = pd.read_pickle(label_input)
        if 'file' in label_df.columns:
            label_df['id'] = [_file[:_file.rfind('.')] for _file in label_df['file']]
            label_df.set_index('id', inplace=True)
            pairwise_labels = label_df.loc[pairwise_ids][label_key].to_numpy()
        else:
            logger.error(f'No "file" column in label_input: {label_input}!!!')        

   # delete rows and cols given by irows_with_NaNs (which may be empty)
    logger.info(f'Removing {len(irows_with_NaNs)} rows with NaNs, leaving {len(pairwise_mat) - len(irows_with_NaNs)} rows...')
    if pairwise_labels is not None:
        manifold_labels = np.delete(pairwise_labels, irows_with_NaNs, axis=0)
    else:
        manifold_labels = None
    manifold_idx = np.delete(pairwise_idx, irows_with_NaNs, axis=0)
    manifold_ids = np.delete(pairwise_ids, irows_with_NaNs, axis=0)
    manifold_mat = np.delete(pairwise_mat, irows_with_NaNs, axis=0)
    manifold_mat = np.delete(manifold_mat, irows_with_NaNs, axis=1)
    
    if sim2dist:
        logger.info('Converting similarity to distance...')
        manifold_mat = 1.0 - manifold_mat

    # check negative values
    num_negs = (manifold_mat < 0).sum()
    if num_negs > 0:
        logger.warning(f'Found {num_negs} negative values in the matrix!!!')
    else:
        logger.info('No negative values found in the matrix.')

    # cluster the pairwise matrix
    algorithm = algorithm.lower()
    if algorithm == 'umap': # UMAP algorithm #
        manifold_args = dict(
            n_neighbors=kwargs.get('n_neighbors', 30), 
            min_dist=kwargs.get('min_dist', 0.15), 
        )

        manifold_fn = umap.UMAP(
            n_components=kwargs.get('n_components', 2),
            metric=kwargs.get('metric', 'precomputed'),
            **manifold_args)

        manifold_out = manifold_fn.fit(manifold_mat)
    elif algorithm == 'tsne': # TSNE algorithm #
        # Notes:
        #  n_features ideally below 50
        #   - perplexity: The number of nearest neighbors to use for approximation, scale as sqrt(n_samples).
        #                 Larger datasets usually require a larger perplexity. 
        #                 Consider selecting a value between 5 and 50. 
        #                 The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
        
        #   - n_iter: Maximum number of iterations for the optimization. A rule of thumb is to set n_iter to be at least 4 times the perplexity.
        #             One way to check is that the largest distance between data points is on the order of ~100
        #   - early_exaggeration: Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.
        #   - n_iter_without_progress: Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration.
        
        manifold_args = dict(
            perplexity=kwargs.get('perplexity', 30.0),  # 30.0
            n_iter=kwargs.get('n_iter', 1000),
        )

        # create a TSNE object with precomputed distance
        manifold_fn = manifold.TSNE(
            n_components=kwargs.get('n_components', 2),
            metric=kwargs.get('metric', 'precomputed'),
            init=kwargs.get('init', 'random'),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1), # -1 has no effect for precomputed distances
            early_exaggeration=kwargs.get('early_exaggeration', 12.0),
            learning_rate=kwargs.get('learning_rate', 'auto'), # usually in the range [10.0, 1000.0].
            n_iter_without_progress=kwargs.get('n_iter_without_progress', 300),
            **manifold_args)

        manifold_out = manifold_fn.fit(manifold_mat)

    elif algorithm == 'mds': # MDS algorithm #
        manifold_args = dict(
            n_init=4,
        )
        manifold_fn = manifold.MDS(
            n_components=2, 
            dissimilarity='precomputed',
            metric=True,
            **manifold_args,
        )

        manifold_out = manifold_fn.fit(manifold_mat)

    else:
        logger.error(f'Unknown clustering algorithm: {algorithm}')
        exit(1)

    pred_embedding = manifold_out.embedding_
    manifold_df = pd.DataFrame({
        'idx': manifold_idx, 
        'id': manifold_ids, 
        'label': manifold_labels, 
        'embedding': list(pred_embedding),
        'hasNaN': [False]*len(pred_embedding)})

    if transform_nans and len(irows_with_NaNs) > 0:
        logger.info(f'Transform {len(irows_with_NaNs)} IDs with NaNs into embedding space...')

        nans_mat = pairwise_mat[irows_with_NaNs, :]
        nans_mat = np.delete(nans_mat, irows_with_NaNs, axis=1)

        # replace NaNs in nans_mat with random values from the rest of the matrix
        nans_mat[np.isnan(nans_mat)] = np.random.choice(pairwise_mat[~np.isnan(pairwise_mat)],
                                    size=nans_mat[np.isnan(nans_mat)].shape[0], replace=True)
        
        if sim2dist:
            nans_mat = 1.0 - nans_mat
        print(nans_mat.shape, pairwise_mat.shape, pairwise_mat[irows_with_NaNs, :].shape)

        # add "(NaN)" to pariwise_labels with NaNs
        pairwise_labels[irows_with_NaNs] = [f'{_label} (NaN)' for _label in pairwise_labels[irows_with_NaNs]]
        
        # this does not work for TSNE
        nans_embeddings = manifold_out.transform(nans_mat) if hasattr(manifold_out, 'transform') \
                     else manifold_out.fit_transform(nans_mat)

        nan_df = pd.DataFrame({
            'idx': irows_with_NaNs, 
            'id': pairwise_ids[irows_with_NaNs], 
            'embedding': list(nans_embeddings),
            'label': pairwise_labels[irows_with_NaNs],
            'hasNaN': [True]*len(irows_with_NaNs),
            })

        manifold_df = pd.concat([manifold_df, nan_df], axis=0)

    save_prefix = f'{save_prefix}_{algorithm}_' + "_".join([f"{k}-{v}" for k, v in manifold_args.items()])
    if save_pkl:
        save_path = save_dir / f'{save_prefix}.pkl'
        logger.info(f'Saving manifold results to {save_path}')
        manifold_df.to_pickle(save_path)

    # plot the embedding
    embeddings = manifold_df['embedding'].to_numpy()
    embeddings = np.vstack(embeddings)

    algorithm = algorithm.upper()
    x = f'{algorithm}-X'
    y = f'{algorithm}-Y'

    kwargs.setdefault('margin_r', 180)
    kwargs.setdefault('marginal_x', 'histogram')
    kwargs.setdefault('marginal_y', 'histogram')
    gdf = glance_df.MyDataFrame(pd.DataFrame({x:embeddings[:, 0], y:embeddings[:, 1], 'label':manifold_df['label']}))
    gdf.plx_xys(x=x, ys=y, color='label', fmt='scatter',
        title=f'Data: {auto_save_prefix}<br>Algo: {algorithm} {gwio.json_str(manifold_args, indent=1, newline=None)}',
        title_font_size=24, 
        xtitle=None, ytitle=None,
        show_img=show_img, show_fig=show_fig,
        save_dir=save_dir, save_prefix=save_prefix,
        **kwargs)

    return manifold_df


def cluster_pairwise_mat(
        inputs=Path.home() / 'database/contarna/strive_2022/nr80-vs-nr80.rnadistance.alnIdentity_pairwise.pkl',
        label_input=Path.home() / 'database/contarna/strive_2022/libset_len30-600.pkl',
                                  # ID matchching label_input['file'].stem against the index of inputs_df,
                                  # which is the stem of queryFile/templFile as specified in slurp_2p.sh
        label_key='moltype',      # the key col to extract from the meta_input
        square_dataframe=False,   # whether to square up the rows and columms of the dataframe
        diagonal=1.0,             # fill diagonal of the matrix before clustering
        diagonal_nan=None,        # fill diagonal NaNs with this value (before converting to distance)
        conjugate_to_nan=True,    # fill asymmetric NaNs with their conjugate values
        symmetric_nan=None,       # fill symmetric NaNs (three None/False numerical value or )
        nan=None,                 # fill all leftover NaNs with this value
        sim2dist=True,            # convert similarity to distance
        algorithm='affinity',     # 'affinity' or 'optics' or 'dbscan' or 'hdbscan' or 'topcut'
        neighbor_recruits=1,      # only used in "topcut"
        adaptive_size=True,       # only used in "topcut"
        assign_outliers=True,     # merge outliers into true clusters
        assign_nans=True,         # merge NaNs into true clusters
        save_dir=None,            # save_dir for the clustering results
        save_prefix=None,         # using the input file name if not specified
        save_pkl=False,
        **kwargs):
    """ cluster a symmetric pariwise matrix with index and columns as IDs """
    args = misc.Struct(locals())
    # logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(inputs, return_save_prefix=True)
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    save_prefix = misc.get_1st_value([
        save_prefix,
        f'{auto_save_prefix}' if auto_save_prefix else None,
        inspect.currentframe().f_code.co_name])
    
    # guess fill_diagonal_nans, fill_diagonal, sim2dist, fit_nans from mat_genre
    mat_genre = auto_save_prefix.lower()
    diagonal_guess = 1.0
    diagonal_nan_guess = 1.0
    fill_nans_guess = 0.0
    fit_nans_guess = False
    sim2dist_guess = True
    
    if 'rnadistance' in mat_genre:
        if 'alnscore' in mat_genre:  # start from 0 and go up to hundreds
            diagonal_nan_guess = 0
            diagonal_guess = 0
            fill_nans_guess = 500   # the larger the more distant
            sim2dist_guess = False
        elif 'seqidentity' in mat_genre or 'alnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0
            sim2dist_guess = True
        elif 'dbnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0
            sim2dist_guess = True
        else:
            logger.error(f'Cannot guess mat preprocessing parameters for mat_genre: {mat_genre}')
    elif 'gardenia' in mat_genre:
        if 'alnscore' in mat_genre:  # range from negative thousands to positive thousands
            diagonal_nan_guess = 0
            diagonal_guess = 0
            fill_nans_guess = 0
            sim2dist_guess = False
        elif 'seqidentity' in mat_genre or 'alnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0
            sim2dist_guess = True
        elif 'dbnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0
            sim2dist_guess = True
        else:
            logger.error(f'Cannot guess mat preprocessing parameters for mat_genre: {mat_genre}')
    elif 'lara2' in mat_genre:
        fit_nans_guess = True
        if 'alnscore' in mat_genre:    # not sure about its meaning, both negative and positive in millions
            diagonal_nan_guess = 0
            diagonal_guess = 0
            fill_nans_guess = 0.0
            sim2dist_guess = False
        elif 'seqidentity' in mat_genre or 'alnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0.0
            sim2dist_guess = True
        else:
            logger.error(f'Cannot guess mat preprocessing parameters for mat_genre: {mat_genre}')        
    elif 'locarna' in mat_genre:
        fit_nans_guess = True
        if 'alnscore' in mat_genre:    # not sure about its meaning, both negative and positive in millions
            diagonal_nan_guess = 0
            diagonal_guess = 0
            fill_nans_guess = 0.0
            sim2dist_guess = False
        elif 'seqidentity' in mat_genre or 'alnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0.0
            sim2dist_guess = True
        else:
            logger.error(f'Cannot guess mat preprocessing parameters for mat_genre: {mat_genre}')               
    elif 'foldalign' in mat_genre:
        fit_nans_guess = True
        if 'alnscore' in mat_genre:    # not sure about its meaning, both negative and positive in millions
            diagonal_nan_guess = 0
            diagonal_guess = 0
            fill_nans_guess = 0.0
            sim2dist_guess = False
        elif 'seqidentity' in mat_genre or 'alnidentity' in mat_genre:
            diagonal_nan_guess = 1.0
            diagonal_guess = 1.0
            fill_nans_guess = 0.0
            sim2dist_guess = True
        else:
            logger.error(f'Cannot guess mat preprocessing parameters for mat_genre: {mat_genre}')               
    else:
        logger.info(f'Cannot guess mat preprocessing parameters for mat_genre: {mat_genre}')

    diagonal_nan = misc.get_1st_value([diagonal_nan, diagonal_nan_guess])
    diagonal = misc.get_1st_value([diagonal, diagonal_guess])
    sim2dist = misc.get_1st_value([sim2dist, sim2dist_guess])

    # extract numpy array from dataframe
    pairwise_idx = np.arange(len(df), dtype=int)
    logger.info(f'Calling bake_pairwise_df() with diagonal_nan: {diagonal_nan}, diagonal: {diagonal}, conjugate_to_nan: {conjugate_to_nan}, symmetric_nan: {symmetric_nan} ...')
    pairwise_mat, pairwise_ids, irows_with_NaNs = bake_pairwise_df(df,
                square_dataframe=square_dataframe,
                diagonal=diagonal, 
                diagonal_nan=diagonal_nan,
                nan=nan,
                conjugate_to_nan=conjugate_to_nan, 
                symmetric_nan=symmetric_nan,
                symmetrize=True,
                save_mat=False, 
                save_pkl=False)
    
    
    # Find the labels for each data point
    pairwise_labels = None
    if label_input is not None and Path(label_input).exists():
        label_df = pd.read_pickle(label_input)
        if 'file' in label_df.columns:
            logger.info(f'Found "file" column in {misc.str_color(label_input)}, will use it to match pairwise labels...')
            label_df['id'] = [Path(_file).stem for _file in label_df['file']]
            label_df.set_index('id', inplace=True)
            pairwise_labels = label_df.loc[pairwise_ids][label_key].to_numpy()
        else:
            logger.error(f'Cannot find "file" column in {label_input}')

    # delete rows and cols given by irows_with_NaNs (which may be empty)
    logger.info(f'Removing {len(irows_with_NaNs)} rows with NaNs, leaving {len(pairwise_mat) - len(irows_with_NaNs)} rows...')
    if pairwise_labels is not None:
        cluster_labels = np.delete(pairwise_labels, irows_with_NaNs, axis=0)
    else:
        cluster_labels = None
    cluster_no = np.delete(pairwise_idx, irows_with_NaNs, axis=0)
    cluster_ids = np.delete(pairwise_ids, irows_with_NaNs, axis=0)
    cluster_mat = np.delete(pairwise_mat, irows_with_NaNs, axis=0)
    cluster_mat = np.delete(cluster_mat, irows_with_NaNs, axis=1)

    if sim2dist:
        logger.info('Converting similarity to distance by subtracting each row by the diagonal element...')
        cluster_mat = cluster_mat.diagonal()[:, np.newaxis] - cluster_mat

    # check negative values
    num_negs = (cluster_mat < 0).sum()
    if num_negs > 0:
        logger.warning(f'Found {num_negs} negative values in the matrix!!!')
    else:
        logger.info('No negative values found in the matrix.')

    algorithm = algorithm.lower()
    if algorithm == 'optics': # OPTICS algorithm #
        cluster_args = dict(
            max_eps=kwargs.get('max_eps', 0.7), # the maximum distance between two samples for them to be considered as in the same neighborhood
            min_samples = kwargs.get('min_samples', 7),  # the number of samples in a neighborhood for a point to be considered as a core point 
            cluster_method=kwargs.get('cluster_method', 'dbscan'), # 'dbscan' or 'xi' or 'hdbscan'
        )
        logger.info(f'Algorithm: {algorithm}, cluster_args: {cluster_args} ...')
        
        cluster_fn = cluster.OPTICS(
            metric='precomputed', # 'minkowski'
            n_jobs=kwargs.get('n_jobs', -1),
            **cluster_args)

        cluster_out = cluster_fn.fit(cluster_mat)

    elif algorithm == 'affinity': # Affinity Propagation algorithm #
        cluster_args = dict(
            damping=kwargs.get('damping', 0.7),  # [0.5, 1.0)
            preference=kwargs.get('preference', 0.1), #-0.5,
        )
        logger.info(f'Algorithm: {algorithm}, cluster_args: {cluster_args} ...')
        cluster_fn = cluster.AffinityPropagation(
            random_state=42,
            affinity='precomputed',
            max_iter=200, 
            convergence_iter=15,  # [10, 15] 
            verbose=True,
            **cluster_args)
        cluster_out = cluster_fn.fit(cluster_mat)

    elif algorithm == 'mds': # MDS algorithm #
        cluster_args = dict(
            n_init=4,
        )
        logger.info(f'Algorithm: {algorithm}, cluster_args: {cluster_args} ...')
        cluster_fn = manifold.MDS(
            n_components=2, 
            dissimilarity='precomputed',
            metric=True,
            **cluster_args
        )

        cluster_out = cluster_fn.fit(cluster_mat)

    elif algorithm == 'topcut': # #
        cluster_args = dict(
            max_eps=kwargs.get('max_eps', 0.5), # the maximum distance between two samples for them to be considered as in the same neighborhood
            min_samples = kwargs.get('min_samples', 23),  # the number of samples in a neighborhood for a point to be considered as a core point 
            neighbor_recruits=neighbor_recruits, # number of neighbor searches to recruit new members
            adaptive_size=adaptive_size, # whether to adaptively change the cluster size
            # cluster_method=kwargs.get('cluster_method', 'dbscan'), # 'dbscan' or 'xi' or 'hdbscan'
        )
        logger.info(f'Algorithm: {algorithm}, cluster_args: {cluster_args} ...')
        
        cluster_out = misc.Struct()
        cluster_out.labels_ = np.full((len(cluster_mat),), -1, dtype=int)

        cluster_next = 0
        indices2mat = np.arange(len(cluster_mat), dtype=int)
        isa_cluster = cluster_mat < cluster_args['max_eps']
        while True:
            cluster_sizes = isa_cluster.sum(axis=1)
            if adaptive_size:
                min_samples = min([cluster_args['min_samples'], max([7, int(len(isa_cluster)*0.07)])])
            else:
                min_samples = cluster_args['min_samples']
            logger.info(f'Extracting cluster: {cluster_next} with total samples: {len(isa_cluster)}, min_samples: {min_samples} ...')
            
            # find the number of values greater than max_eps in each row
            cluster_sizes[cluster_sizes < min_samples] = 0
            if cluster_sizes.sum() == 0:
                logger.info(f'No more clusters with sizes > {min_samples} found with {len(isa_cluster)} samples left!')
                break

            cluster_means = (cluster_mat[indices2mat, indices2mat] * isa_cluster).sum(axis=1) / (cluster_sizes + 1e-6)
            # cluster_means[np.isnan(cluster_means)] = np.inf
            # cluster_means[cluster_means < 0.1] = 0.1

            icenter = np.argmax(cluster_sizes)
            equiv_centers = np.where(cluster_sizes == cluster_sizes[icenter])[0]
            icenter = equiv_centers[np.argmin(cluster_means[equiv_centers])]

            cluster_new_indices = np.where(isa_cluster[icenter])[0]
            # num_neighbor_searches = min([neighbor_recruits, 0.777 / cluster_args['max_eps']])
            for _ in range(neighbor_recruits):
                # can be more efficient if adding new neighbors incrementally
                with_neighbors = np.where(isa_cluster[cluster_new_indices])[1]
                with_neighbors = np.unique(with_neighbors)
                num_recruits = len(with_neighbors) - len(cluster_new_indices)
                logger.info(f'Found {num_recruits:7d} new neighbors, add to cluster: {cluster_next} of size {len(cluster_new_indices)}...')

                if num_recruits > 0:
                    cluster_new_indices = with_neighbors
                else:
                    break

            # one can turn on the greedy mode

            # assign the cluster number
            cluster_out.labels_[indices2mat[cluster_new_indices]] = cluster_next
            logger.info(f'Cluster {cluster_next} has {cluster_sizes[icenter]} members.')

            # cut the cluster out of the matrix
            isa_cluster = np.delete(isa_cluster, cluster_new_indices, axis=0)
            isa_cluster = np.delete(isa_cluster, cluster_new_indices, axis=1)
            indices2mat = np.delete(indices2mat, cluster_new_indices, axis=0)
            
            cluster_next += 1
    else:
        logger.error(f'Unknown clustering algorithm: {algorithm}')
        exit(1)

    # summarize the clustering results
    pred_labels = np.array(cluster_out.labels_)
    i_outliers = np.where(pred_labels == -1)[0]
    num_outliers = len(i_outliers)
    num_clusters = len(np.unique(pred_labels)) - (1 if num_outliers else 0)
    print(f'Predicted number of clusters: {num_clusters}')
    print(f'Predicted number of outliers: {num_outliers}')

    # convert the array of strings to an array of integers
    unique_labels = np.unique(cluster_labels)
    label2int = {unique_labels[i]: i for i in range(len(unique_labels))}
    true_labels = np.array([label2int[_s] for _s in cluster_labels])
    # true_labels = np.array([np.where(unique_labels == _s)[0][0] for _s in cluster_labels])
    print("                 Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, pred_labels))
    print("                Completeness: %0.3f" % metrics.completeness_score(true_labels, pred_labels))
    print("                   V-measure: %0.3f" % metrics.v_measure_score(true_labels, pred_labels))
    print("         Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(true_labels, pred_labels))
    print(
          " Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(true_labels, pred_labels)
    )
    if len(pred_labels) > num_clusters > 1:
        print(
            "      Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(cluster_mat, pred_labels, metric="precomputed")
        )
    else:
        logger.critical('One or no cluster found. Silhouette Coefficient is undefined!')
        # exit(1)

    if assign_outliers and num_outliers > 0:
        logger.info(f'Merging {num_outliers} outliers into the list of NaNs...')
        # combine the outliers with the list of IDs with NaNs
        idx_to_assign = cluster_no[i_outliers]
        cluster_df = pd.DataFrame({
            'idx': np.delete(cluster_no, i_outliers, axis=0),
            'id': np.delete(cluster_ids, i_outliers, axis=0), 
            'label': None if cluster_labels is None else np.delete(cluster_labels, i_outliers, axis=0), 
            'cluster': np.delete(pred_labels, i_outliers, axis=0),
            'hasNaN': [False]*(len(pred_labels) - num_outliers),
            'isaOutlier': [False]*(len(pred_labels) - num_outliers),
            })        
    else:
        idx_to_assign = np.empty((0), int)
        cluster_df = pd.DataFrame({
            'idx': cluster_no, 
            'id': cluster_ids, 
            'label': cluster_labels, 
            'cluster': pred_labels,
            'hasNaN': [False]*len(pred_labels),
            'isaOutlier': [False]*len(pred_labels),
            })
        if num_outliers > 0:
            cluster_df.loc[i_outliers, 'isaOutlier'] = True

    # force dtypes of cluster_df: idx, cluster, hasNaN are integers
    cluster_df['idx'] = cluster_df['idx'].astype(int)
    cluster_df['id'] = cluster_df['id'].astype(str)
    cluster_df['label'] = cluster_df['label'].astype(str)
    cluster_df['cluster'] = cluster_df['cluster'].astype(int)
    cluster_df['hasNaN'] = cluster_df['hasNaN'].astype(bool)
    
    dfs_to_merge = [] # merge to the cluster_df in the end
    if assign_nans and len(irows_with_NaNs) > 0:
        idx_to_assign = np.append(idx_to_assign, irows_with_NaNs)
    elif len(irows_with_NaNs) > 0: # append the entries with NaNs, but with cluster=min_cluster-1
        dfs_to_merge.append(pd.DataFrame({
            'idx': irows_with_NaNs, 
            'id': pairwise_ids[irows_with_NaNs], 
            'cluster': [cluster_df['cluster'].min()-1]*len(irows_with_NaNs), 
            'label': None if pairwise_labels is None else pairwise_labels[irows_with_NaNs],
            'hasNaN': [True]*len(irows_with_NaNs),
            'isaOutlier': [True]*len(irows_with_NaNs),
            }))

    if len(idx_to_assign) > 0:
        logger.info(f'Assigning {len(idx_to_assign)} IDs to clusters...')

        if len(cluster_df) > 0:
            # group cluster_df by cluster and get the list of idx for each group
            cluster_grps = cluster_df.groupby('cluster').agg({'idx': lambda x: list(x),
                                                            'id': lambda x: list(x),
                                                            'label': lambda x: list(x),
                                                            'hasNaN': lambda x: list(x)})
            cluster_grps = cluster_grps.reset_index()

            # assign cluster numbers to ids with NaNs by finding the closest cluster (use the max similarity to the cluster points)
            similarity_to_clusters = []
            mat_2_assign = np.nan_to_num(pairwise_mat[idx_to_assign, :], nan=0.0)
            for i in range(len(cluster_grps)):
                similarity_to_clusters.append(np.max(mat_2_assign[:, cluster_grps['idx'][i]], axis=1))
                
            similarity_to_clusters = np.stack(similarity_to_clusters, axis=1)

            # find the closest cluster for each row with NaNs
            closest_cluster = np.argmax(similarity_to_clusters, axis=1)
            cluster_to_assign = cluster_grps['cluster'][closest_cluster].values
        else:
            logger.warning(f'No clusters found. Assigning all {len(idx_to_assign)} IDs to cluster -1.')
            cluster_to_assign = [-1] * len(idx_to_assign)

        # add the rows and cols with NaNs
        dfs_to_merge.append(pd.DataFrame({
            'idx': idx_to_assign,
            'id': pairwise_ids[idx_to_assign], 
            'cluster': cluster_to_assign, # [min(labels)-1]*len(ids_with_NaNs), 
            'label': None if pairwise_labels is None else pairwise_labels[idx_to_assign], 
            'hasNaN': [_i in irows_with_NaNs for _i in idx_to_assign],
            'isaOutlier': [True]*len(idx_to_assign),
            }))

    # combine the dataframes, save and plot
    if len(dfs_to_merge):
        cluster_df = pd.concat([cluster_df, *dfs_to_merge], axis=0)

    save_prefix = f'{save_prefix}_{algorithm}_' + "_".join([f"{k}-{v}" for k, v in cluster_args.items()])
    if save_pkl:
        save_path = f'{save_prefix}.pkl'
        if save_dir is not None:
            save_path = os.path.join(save_dir, save_path)
            os.makedirs(save_dir, exist_ok=True)
            
        logger.info(f'Saving clustering results to {save_path}')
        cluster_df.to_pickle(save_path)

    # get the percentage of each label in cluster_df
    label_counts = cluster_df['label'].value_counts()
    label_counts = label_counts.to_dict()
    cluster_df['label'] = cluster_df['label'].apply(lambda x: f'{x}<br>({label_counts[x]:,d}; {label_counts[x]/len(cluster_df)*100:.1f}%)' if x is not None else None)

    # plot the clustering results
    # get the list of "label" values and their counts
    label_counts = cluster_df['label'].value_counts().to_dict()

    gdf = glance_df.MyDataFrame(cluster_df)
    gdf.plx_xys(x='cluster', ys=None, facet_row='label', 
               color='isaOutlier' if num_outliers > 0 and assign_outliers else None,
               color_discrete_map={True:'red', False:'blue'},
               fmt='hist',
               category_orders={'label': sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)},
               nbins=cluster_df['cluster'].max() - cluster_df['cluster'].min() + 1,
               showlegend=False,
               xtitle=None, #'Cluster No.',
               ytitle=None, # 'Count',
               title=f'{auto_save_prefix} ({len(cluster_df)}), nan_rows: {len(irows_with_NaNs)}, outliers: {num_outliers}<br>Algo: {algorithm} {gwio.json_str(cluster_args, indent=1, newline=None)}',
            #    title_font_size=24,
               save_dir=save_dir,
               save_prefix=f'{save_prefix}',
               **kwargs)
               
    return cluster_df        


def bake_pairwise_df(
        data_names,
        square_dataframe=True,    # square up the columns and rows of the dataframe
        diagonal=None, 
        diagonal_nan=None,
        nan=None,
        conjugate_to_nan=True,
        symmetric_nan=None,
        symmetrize=True,          # symmetrize the matrix
        save_pkl=False,
        save_mat=False,
        save_prefix=None,
        **kwargs):
    """ fix up a dataframe containing pairwise matrix (usually after pivoting two dataframe cols)
            1) square up the matrix by filling NaNs
            2) fill diagonal NaNs if passed
            3) overwrite diagonal values if passed
            4) and remove asymmetric NaNs for clustering
    """
    args = misc.Struct(locals())
    # logger.debug('Arguments:\n' + gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([
        kwargs.get('save_prefix', None),
        f'{auto_save_prefix}_pairwise' if auto_save_prefix else None,
        inspect.currentframe().f_code.co_name])

    if square_dataframe:
        # get index and column unions and intersections
        indices = set(df.index.to_list())
        logger.info(f'Number of rows: {len(df)}, number of unique indices: {len(indices)}')
        columns = set(df.columns.to_list())
        logger.info(f'Number of cols: {df.shape[1]}, number of unique columns: {len(columns)}')

        allcols = list(indices.union(columns))
        logger.info(f'Number of unique indices and columns: {len(allcols)}')
        commons = list(indices.intersection(columns))
        logger.info(f'Number of common indices and columns: {len(commons)}')

        # commons = [_s for _s in commons if _s not in [np.nan, None, '']]
        # df = df.loc[commons, commons]

        # create a new squared dataframe with the union of indices and columns
        logger.info(f'Create a new dataframe with the union of indices and columns...')
        df = df.reindex(index=allcols, columns=allcols, fill_value=np.nan)

    pairwise_mat, irows_with_NaNs = bake_pairwise_mat(df.values,
                diagonal=diagonal,
                diagonal_nan=diagonal_nan,
                nan=nan,
                conjugate_to_nan=conjugate_to_nan, 
                symmetric_nan=symmetric_nan,
                symmetrize=symmetrize,
                save_mat=save_mat, 
                save_prefix=save_prefix,
                **kwargs)

    if save_pkl:
        df.loc[:, :] = pairwise_mat
        df.to_pickle(f'{save_prefix}.pkl')

    return pairwise_mat, df.index, irows_with_NaNs


    # fill diagonals or just the NaNs in the diagonal
    if diagonal is not None:
        logger.info(f'Filling diagonal with {diagonal}')
        np.fill_diagonal(df.values, diagonal)
    elif diagonal_nan is not None:
        diag_vals = df.values.diagonal()
        isa_diag_NaNs = np.isnan(diag_vals)
        num_diag_NaNs = isa_diag_NaNs.sum()
        logger.info(f'Number of NaNs in the diagonal: {num_diag_NaNs}, to be filled with {diagonal_nan}')

        idx_diag_NaNs = np.where(isa_diag_NaNs)[0]
        df.iloc[idx_diag_NaNs, idx_diag_NaNs] = diagonal_nan
    else:
        logger.info(f'Not filling diagonal as both fill_diagonal and fill_diagonal_nans are None')

    # check for NaNs and try to fill the NaNs with their conjugates
    if conjugate_to_nan or symmetric_nan is not None:
        logger.info('Checking for NaNs in the matrix...')
        isa_NaNs = df.isna().values.astype(int)
        num_NaNs = isa_NaNs.sum()
        logger.info(f'Number of NaNs: {num_NaNs} in the squared dataframe of shape: {df.shape}')

        # find the number of elements in isa_NaNs for which the conjugate is not NaN
        isa_symmetric_NaNs = np.transpose(isa_NaNs) * isa_NaNs
        num_symmetric_NaNs = isa_symmetric_NaNs.sum()
        logger.info(f'Number of symmetric NaNs: {num_symmetric_NaNs} in the squared dataframe of shape: {df.shape}')

        num_NaNs_per_row = isa_symmetric_NaNs.sum(axis=1)
        irows_with_NaNs = np.where(num_NaNs_per_row > 0)[0]
        logger.info(f'Number of row/cols with symmetric NaNs: {len(irows_with_NaNs)}')
        logger.info(f'Number of NaNs in a row/col, min: {num_NaNs_per_row.min()}, max: {num_NaNs_per_row.max()}')
    else:
        irows_with_NaNs = np.array([], dtype=int)
        
    # fill asymmetric NaNs with their conjugates
    if conjugate_to_nan and num_NaNs > num_symmetric_NaNs:
        logger.info(f'Fill conjugate values to asymmetric NaNs (count: {num_NaNs - num_symmetric_NaNs})')
        df.fillna(0.0, inplace=True)
        df = df + np.transpose(df.values) * isa_NaNs

    # convert df to a symmetric matrix
    if symmetrize:
        logger.info('Symmetrizing the matrix...')
        df = (df + np.transpose(df.values)) / 2.0
        # df.values = (df.values + np.transpose(df.values)) / 2.0
        
    # fill or restore the symmetric NaNs
    if symmetric_nan is not None and num_symmetric_NaNs > 0:
        logger.info(f'Fill the symmetric NaNs (count: {num_symmetric_NaNs}) with {symmetric_nan}')
        df[isa_symmetric_NaNs.astype(bool)] = symmetric_nan
        irows_with_NaNs = np.array([], dtype=int)
        # todo: consider to fill with random numbers between 0 and 1 symmetrically
    elif conjugate_to_nan and num_symmetric_NaNs > 0:
        logger.info(f'Restore the symmetric NaNs (count: {num_symmetric_NaNs})')
        df[isa_symmetric_NaNs.astype(bool)] = np.nan

    if save_mat:
        mat_file = f"{kwargs['save_prefix']}.mat"
        logger.info(f'Saving the pairwise matrix to {misc.str_color(mat_file)}...')
        np.savetxt(mat_file, pairwise_mat, fmt='%6.4f')

    return pairwise_mat, df.index, irows_with_NaNs


def bake_pairwise_mat(
        pairwise_mat,             # a square pairwise matrix
        diagonal=None,            # fill diagonal with this value if not None or False
        diagonal_nan=None,        # fill diagonal NaNs (if not None or False), negative value: random value from 0 to abs(diagonal_nan), positive value: as-is
        nan=None,                 # fill all NaNs with this value if not None or False
        conjugate_to_nan=True,    # fill NaNs with their non-nan conjugates
        symmetric_nan=None,       # fill symmetric NaNs (if not None or False), negative value: random value from 0 to abs(symmetric_nan), positive value: as-is
        symmetrize=True,          # symmetrize the matrix
        save_mat=False,
        save_prefix="pairwise_mat",
        **kwargs):
    """ fix up a square pairwise matrix
            1) fill diagonal NaNs if passed
            2) overwrite diagonal values if passed
            3) and remove asymmetric NaNs for clustering
    """
    assert pairwise_mat.shape[0] == pairwise_mat.shape[1], 'pairwise_mat is not a square matrix'

    # fill diagonals or just the NaNs in the diagonal
    if diagonal not in [None, False]:
        logger.info(f'Filling diagonal with {diagonal}')
        np.fill_diagonal(pairwise_mat, diagonal)
    elif diagonal_nan not in [None, False]:
        diag_vals = pairwise_mat.diagonal()
        isa_diag_NaNs = np.isnan(diag_vals)
        num_diag_NaNs = isa_diag_NaNs.sum()
        logger.info(f'Number of NaNs in the diagonal: {num_diag_NaNs} ({num_diag_NaNs/len(isa_diag_NaNs)*100:.4f}%), to be filled with {diagonal_nan}')

        idx_diag_NaNs = np.where(isa_diag_NaNs)[0]
        if diagonal_nan < 0:
            logger.info(f'Filling diagonal NaNs with random numbers between 0 and {abs(diagonal_nan)}...')
            pairwise_mat[idx_diag_NaNs, idx_diag_NaNs] = np.random.rand(num_diag_NaNs) * abs(diagonal_nan)
        else:
            pairwise_mat[idx_diag_NaNs, idx_diag_NaNs] = diagonal_nan
    else:
        logger.info(f'Not filling diagonal as both fill_diagonal and fill_diagonal_nans are None')

    # check for NaNs only if necessary
    if conjugate_to_nan or symmetric_nan not in [None, False] or nan not in [None, False]:
        logger.info('Checking for NaNs in the matrix...')
        isa_NaNs = np.isnan(pairwise_mat)
        num_NaNs = isa_NaNs.sum()
        logger.info(f'Number of NaNs: {num_NaNs} ({num_NaNs/isa_NaNs.size*100:.5f}%) in the squared dataframe of shape: {pairwise_mat.shape}')

        # find the number of elements in isa_NaNs for which the conjugate is not NaN
        isa_symmetric_NaNs = np.logical_and(np.transpose(isa_NaNs), isa_NaNs)
        num_symmetric_NaNs = isa_symmetric_NaNs.sum()
        logger.info(f'Number of symmetric NaNs: {num_symmetric_NaNs} ({num_symmetric_NaNs/isa_NaNs.size*100:.5f}%) in matrix shape: {pairwise_mat.shape}')

        num_NaNs_per_row = isa_symmetric_NaNs.sum(axis=1)
        irows_with_NaNs = np.where(num_NaNs_per_row > 0)[0]
        logger.info(f'Number of row/cols with symmetric NaNs: {len(irows_with_NaNs)}')
        logger.info(f'Number of NaNs in any row/col, min: {num_NaNs_per_row.min()}, max: {num_NaNs_per_row.max()}')
    else:
        irows_with_NaNs = np.array([], dtype=int)

    # fill the NaNs
    if nan not in [None, False] and num_NaNs > 0:
        logger.info(f'Fill the leftover NaNs (count: {num_NaNs}) with {nan}')
        if nan < 0:
            logger.info(f'Filling NaNs with random numbers between 0 and {abs(nan)}...')
            pairwise_mat[isa_NaNs] = np.random.rand(num_NaNs) * abs(nan)
        else:
            np.nan_to_num(pairwise_mat, copy=False, nan=nan)
        irows_with_NaNs = np.array([], dtype=int)

    # fill asymmetric NaNs with their conjugates
    if conjugate_to_nan and num_NaNs > num_symmetric_NaNs:
        logger.info(f'Fill conjugate values to asymmetric NaNs (count: {num_NaNs - num_symmetric_NaNs})')
        isa_asymmetric_NaNs = np.logical_and(isa_NaNs, ~isa_symmetric_NaNs)
        pairwise_mat[isa_asymmetric_NaNs] = pairwise_mat[np.transpose(isa_asymmetric_NaNs)]
        
    # fill symmetric NaNs
    if symmetric_nan not in [None, False] and num_symmetric_NaNs > 0:
        irows_with_NaNs = np.array([], dtype=int)
        logger.info(f'Fill the symmetric NaNs (count: {num_symmetric_NaNs}) with {symmetric_nan}')
        if symmetric_nan < 0:
            logger.info(f'Filling symmetric NaNs with random numbers between 0 and {abs(symmetric_nan)}...')
            pairwise_mat[isa_symmetric_NaNs] = np.random.rand(num_symmetric_NaNs) * abs(symmetric_nan)
        else:
            pairwise_mat[isa_symmetric_NaNs] = symmetric_nan

    # find the minimal set of rows for removing all NaNs
    if len(irows_with_NaNs):
        # gradually remove NaNs from the matrix
        isa_NaNs = np.isnan(pairwise_mat)
        num_NaNs_per_row = isa_NaNs.sum(axis=1)
        irows_to_delete = []
        while True:
            irow_max_NaNs = np.argmax(num_NaNs_per_row)
            max_NaNs = num_NaNs_per_row[irow_max_NaNs]
            logger.info(f'Found {max_NaNs} NaNs in row {irow_max_NaNs}...')
            if max_NaNs > 0:
                irows_to_delete.append(irow_max_NaNs)
                num_NaNs_per_row -= isa_NaNs[:, irow_max_NaNs] # delete that col (symmetric to row)
                num_NaNs_per_row[irow_max_NaNs] = 0
            else:
                logger.info(f'No more NaNs found in the matrix: num of rows with NaNs: {len(irows_with_NaNs)}, number of rows to delete: {len(irows_to_delete)}!')
                print(irows_to_delete)
                break
        irows_with_NaNs = np.array(irows_to_delete, dtype=int)

    # convert df to a symmetric matrix
    if symmetrize:
        logger.info('Symmetrizing the matrix...')
        pairwise_mat = (pairwise_mat + np.transpose(pairwise_mat)) / 2.0

    if save_mat:
        mat_file = f"{save_prefix}.npy"
        logger.info(f'Saving the pairwise matrix to {misc.str_color(mat_file)}...')
        # np.savetxt(mat_file, pairwise_mat, fmt='%6.4f')
        np.save(mat_file, pairwise_mat, allow_pickle=True, fix_imports=False)

    return pairwise_mat, irows_with_NaNs

    
def bake_bert_vocab(
        letters='ACGT',    # letters used for making kmer
        k=3,               # k-mer
        padding=None,      # padding characeter (e.g., N)
        **kwargs):
    """ generate bert vocab file for kmer """

    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    if padding:
        if k % 2 != 1:
            logger.warning(f'k: {k} must be odd with padding!!!')
        nn = k // 2
        for num_pads in range(nn, 0, -1):
            pads = padding * num_pads
            for word in itertools.product(letters, repeat=k - num_pads):
                word = ''.join(word)
                vocab.append(pads + word)
                vocab.append(word + pads)

    for word in itertools.product(letters, repeat=k):
        vocab.append(''.join(word))

    save_path = f'bert{k}{padding.lower() if padding else ""}_vocab.txt'
    logger.info(f'Generated a total of {len(vocab)} words')
    logger.info(f'Saving file: {save_path} ...')
    with open(save_path, 'w') as iofile:
        iofile.writelines('\n'.join(vocab))


def bake_bert_chunks(*data_names, save_dir='./',        # dataframe or dict (csv or pkl)
                     save_prefix=None,                 # save to {save_prefix}_chunks.fasta
                     convert2upper=True,               # conver to upper case
                     convert2dna=True,                 # convert to DNA
                     convert2cdna=False,               # convert to cDNA if id endswith('-')
                     vocab='ACGT',                     # the vocabulary set of characters
                     method='contig',                  # method for dividing chunks: contig/sample
                     coverage=2.0,                     # coverage only if method is sample
                     drop_last=False,                  # whether to drop the leftover if <min_len
                     min_len=30, max_len=510,          # chunk min and max lengths
                     num_cpus=0.3,                     # number of CPUs to use for multiprocessing
                     tqdm_disable=False,               # not yet implemented
                     **kwargs):
    """ start from a dataframe of sequences and chunkerize each seq for bert mlm"""

    save_dir = Path(save_dir)
    args = misc.Struct(locals())
    logger.info('Arguments:')
    print(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    args.auto_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'noname'])

    if convert2upper:
        logger.info('Converting all sequences to upper case...')
        df['seq'] = df['seq'].str.upper()

    if convert2dna:
        df['seq'] = misc.mpi_map(
            functools.partial(molstru.seq2DNA, vocab_set=set(vocab), verify=True),
            tqdm(df.seq, desc='Seq2DNA'),
            num_cpus=num_cpus,
            quiet=True)

    if convert2cdna:
        df['id'] = df['id'].str.strip()
        is_minus = df['id'].str.endswith('-')
        num_minus_strands = is_minus.sum()
        logger.info(f'Found {num_minus_strands} minus strand sequences')
        if num_minus_strands:
            df.loc[is_minus, 'seq'] = misc.mpi_map(
                functools.partial(molstru.seq2cDNA, reverse=True, verify=False),
                tqdm(df[is_minus].seq, desc='Seq2cDNA'),
                num_cpus=num_cpus, quiet=True)

    # chunkerize
    fn_seq2chunks = functools.partial(molstru.seq2chunks, vocab_set=set(vocab),
            method=method, min_len=min_len, max_len=max_len,
            drop_last=drop_last, coverage=coverage)

    logger.info(f'Chunkerize {len(df)} sequences...')
    chunk_seqs = misc.mpi_map(fn_seq2chunks, tqdm(df.seq, desc='Seq2Chunk'),
            num_cpus=num_cpus, quiet=True)

    assert len(chunk_seqs) == len(df), 'Sequence count changed after seq2chunks'

    # chunk_seqs = misc.unpack_list_tuple(chunk_seqs)
    # logger.info(f'Generated a total of {len(chunk_seqs)} chunks')

    # save to fasta
    args.save_path = save_dir / (args.auto_prefix + f'_chunks{min_len}-{max_len}_{method}.fasta')
    logger.info(f'Saving chunks to fasta file: {args.save_path}')
    args.num_chunks = 0
    with args.save_path.open('w') as iofile:
        # for i, id in tqdm(enumerate(df.id), desc='Save fasta'):
        for i in tqdm(range(len(df)), desc="Save fasta"):
            id = df.iloc[i].id
            args.num_chunks += len(chunk_seqs[i])
            if len(chunk_seqs[i]) == 0:
                logger.info(f'No chunks for id: {id}, len: {df.iloc[i]["len"]}, seq: {df.iloc[i].seq}')
                continue
            for i, seq in enumerate(chunk_seqs[i]):
                iofile.writelines([f'>{id}|chunk{i + 1}\n', seq, '\n'])

    logger.info(f'Saved a total of {args.num_chunks} chunks.')
    gwio.dict2json(args.__dict__, fname=args.save_path.with_suffix('.json'))


def add_bert_encoding(data_names,
        model_path=Path.home() / 'workbench/bertarna/ncRNA5n_l12c256/checkpoint-200000',
        k=5,
        padding='N',
        max_len=509,
        batch_size=4,
        save_prefix=None,
        **kwargs):
    """ add bert encoder representation and attention matrix to midat """
    args = misc.Struct(locals())
    logger.info('Arguments:\n' + gwio.json_str(args.__dict__))

    # load sequence data
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value((save_prefix, auto_save_prefix, 'noname'))

    if max_len is not None:
        logger.info(f'Removing sequences longer than {max_len}...')
        df['len'] = df.seq.str.len()
        df = df[df['len'] <= max_len]

    # load model
    import torch
    from transformers import BertModel, BertTokenizer, DNATokenizer

    tokenizer_name = f'dna{k}' + padding.lower() if padding is not None else ''
    model_path = args.model_path
    model = BertModel.from_pretrained(model_path,
            output_attentions=True, output_hidden_states=True)
    tokenizer = DNATokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

    # calculate
    num_seqs = len(df)
    iseq = df.columns.get_loc('seq')
    bert_dataset, attn_dataset = [], []
    for ibatch, istart in tqdm(list(enumerate(range(0, num_seqs, batch_size)))):
        # use iloc here because index is no longer continuous
        batch_seqs = df.iloc[istart:istart + batch_size, iseq]
        batch_seqs = [molstru.seq2DNA(_s, verify=True) for _s in batch_seqs]
        batch_seqs = [_s.upper() for _s in batch_seqs]
        batch_lens = [len(_s) for _s in batch_seqs]

        batch_sentences = [molstru.seq2kmer(_s, k=5, padding='N') for _s in batch_seqs]
        # if len(batch_sentences) == 1:
        #     batch_sentences = batch_sentences[0]

        if isinstance(batch_sentences, str):
            # one sentence only, no attention mask needed
            inputs = tokenizer.encode_plus(batch_sentences, sentence_b=None,
                    return_tensors='pt', add_special_tokens=True,
                    max_length=None,
                    pad_to_max_length=False,
                    )
            attention_mask = None
        else:
            inputs = tokenizer.batch_encode_plus(batch_sentences, sentence_b=None,
                    return_tensors='pt', add_special_tokens=True,
                    max_length=None,
                    pad_to_max_length=True,
                    return_attention_masks=True,
                    )
            attention_mask = inputs['attention_mask']

        # inputs: {'input_ids': [batch_size, max(seqs_len) + 2],
        #          'token_type_ids': [batch_size, max(seqs_len) + 2],
        #          'attention_mask': [batch_size, max(seqs_len) + 2]}
        # CLS is added to the beginning
        # EOS is added to the end

        outputs = model(inputs['input_ids'], attention_mask=attention_mask)

        # outputs[0]: [batch_size, len(tokens)+2, 768] (the final hidden states of the bert base)
        # outputs[1]: [batch_size, 1, 768] (not sure what it is, sentence level summary?)
        # outputs[2]: hidden states
        #   [batch_size, max(seq_len) + 2, 768] * num_layers + 1
        # outputs[3]: attention matrix
        #   a tuple of length=num_layers
        #   each item has shape of [batch_size, nheads, len(tokens)+2, len(tokens)+2]

        # verify output[0] shoud be the same as the last hidden states: ouputs[0] == outputs[2][-1]
        if torch.any(outputs[0] != outputs[2][-1]):
            raise ValueError('Outputs[0] != outputs[2][-1], please check!!!')
        else:
            logger.debug(f'Outputs[0] == outputs[2][-1], good to go!')

        # save bert and attention matrix
        for i in range(len(batch_seqs)):
            bert_dataset.append(outputs[0][i, 1:batch_lens[i] + 1, :].detach().numpy().astype(np.float32))
            # the last layer only
            attn_dataset.append(outputs[-1][-1][i, :, 1:batch_lens[i] + 1, 1:batch_lens[i] + 1].detach().numpy().astype(np.float32))

    channel_size = bert_dataset[0].shape[-1]
    save_file = save_prefix + f'_bert{channel_size}.pkl'
    logger.info(f'Saving pickle file: {save_file} ...')
    df['bert'] = bert_dataset
    df.to_pickle(save_file)

    save_file = save_prefix + f'_attn{channel_size}.pkl'
    df.drop(columns='bert', inplace=True)
    logger.info(f'Saving pickle file: {save_file} ...')
    df['attn'] = attn_dataset
    df.to_pickle(save_file)


def add_cdhit_clusters(data_names,
        clstr_file=None,
        nr_pct=80,         # the redundancy level (x100)
        left_on='id',
        save_prefix=None,
        **kwargs):
    """ add cd-hit-est clustering results as cdhit_nr??_?? columns """
    nr_pct = int(nr_pct)
    args = misc.Struct(locals())
    logger.info('Arguments:\n' + gwio.json_str(args.__dict__))

    # load sequence data
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value((save_prefix, f'{auto_save_prefix}_cdhit{nr_pct}', 'noname'))

    if clstr_file is None:
        clstr_file = f'{auto_save_prefix}_nr{nr_pct}.fasta.clstr'
    if not os.path.isfile(clstr_file):
        logger.error(f'Canot find CDHIT clstr file: {clstr_file} !!!')
        return

    clstr_id = None
    clstr_data = {'id': [], 'clstr_id': [], 'len': [], 'psi': []}
    with open(clstr_file, 'r') as iofile:
        for line in iofile.readlines():
            if line[0] == '>':
                clstr_id = int(line.split()[-1])
            else:
                tokens = line.split()
                if tokens[-1][-1] == '*':
                    clstr_data['psi'].append(100.0)
                elif tokens[-1][-1] == '%':
                    clstr_data['psi'].append(float(tokens[-1][2:-1]))
                else:
                    logger.error(f'Expected either % or * at the end, but got: {line}')
                    continue

                clstr_data['len'].append(int(tokens[1][:-3]))
                clstr_data['id'].append(tokens[2][1:-3])
                clstr_data['clstr_id'].append(clstr_id)

    clstr_data = pd.DataFrame(clstr_data)
    nr_str = f'cdhit_nr{nr_pct}'
    clstr_data.rename(columns={
        'len': f'{nr_str}_len',
        'clstr_id': f'{nr_str}_id',
        'psi': f'{nr_str}_psi'
        }, inplace=True)

    # merge to the parent dataframe, the only problem is that ID may not match
    df = df.merge(clstr_data, how='left',  left_on=left_on, right_on='id',
            suffixes=[None, datetime.now().strftime("_%Y-%m-%d")])
    save_files(df, save_prefix=save_prefix, **kwargs)
    

def parse_blastn_output(data_names,
        blastn_output=None,
        save_prefix=None,
        **kwargs):
    """ parse blastn output and save the list of unique ids in the query and database """
    args = misc.Struct(locals())
    # logger.info('Arguments:\n' + gwio.json_str(args.__dict__))

    df = pd.read_csv(data_names, sep='\t', header=None, 
                    names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])

    if save_prefix is None:
        save_prefix = Path(data_names).stem
    
    logger.info(f'Saving parsed blastn output to {save_prefix}.csv ...')
    df.to_csv(f'{save_prefix}.csv', index=False, header=True)

    for col in ['qseqid', 'sseqid']:
        save_file = f'{save_prefix}.{col}'
        logger.info(f'Saving unique {col} to {save_file} ...')
        df[col].unique().tofile(save_file, sep='\n', format='%s')
        # df_unique = df.drop_duplicates(subset=['qseqid'])
        # df_unique['qseqid'].to_csv('unique.txt', index=False, header=False)


def mutate_sequences(data_names,             # dataframe in csv, pkl
                   rate=None,                # rate for all sequences (stems included!)
                   stem_rate=None,           # rate for stems
                   loop_rate=None,           # rate for loops
                   num_cpus=0.6,             # number/percentage of CPUs
                   save_dir='./',            # current dir is default
                   save_prefix=None,         # various suffixes will be added (overwritten by save_name)
                   save_name=None,           # the actual save_name
                   save_pkl=True,            # save results into pkl
                   save_fasta=True,          # save a single fasta file extracted from results
                   save_csv=False,           # save a summary csv file based on results
                   save_lib=False,           # save various columns into individual files (caution: many files)
                   named_after='file',       # how to name each individual file
                   save_genre=['seq'],       # which types of individual files to save
                   tqdm_disable=False,       # disable tqdm
                   **kwargs):
    """ mutate sequences with options for different rates on stems and loops """

    save_dir = Path(save_dir)
    args = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_prefix = get_midat(data_names, return_save_prefix=True)
    args.auto_prefix = misc.get_1st_value([save_prefix, auto_prefix, 'noname'])

    if rate is not None:
        fn_mutate = functools.partial(molstru.mutate_residues, rate=rate, return_list=True)
        df['seq'] = misc.mpi_map(fn_mutate,
            df.seq if tqdm_disable else tqdm(df.seq, desc='MutateResidues'),
            num_cpus=num_cpus, quiet=True)
        args.auto_prefix += f'_all{rate:0.2f}'

    if stem_rate is not None or loop_rate is not None:
        if 'ct' not in df.columns:
            logger.critical(f'Cannot mutate stem or loop without ct in the dataframe!!!')
        fn_mutate = functools.partial(molstru.mutate_stems_loops, rate=0.0,
                stem_rate=stem_rate, loop_rate=loop_rate, return_list=True)
        df['seq'] = misc.mpi_starmap(fn_mutate,
            zip(df.seq, df.ct) if tqdm_disable else \
            tqdm(zip(df.seq, df.ct), desc='MutateStemLoops'),
            num_cpus=num_cpus, quiet=True)
        if stem_rate is not None: args.auto_prefix += f'_stem{stem_rate:0.2f}'
        if loop_rate is not None: args.auto_prefix += f'_loop{loop_rate:0.2f}'

    if save_pkl or save_fasta or save_csv:
        save_lumpsum_files(df, save_dir=save_dir,
            save_prefix=args.auto_prefix if save_name is None else save_name,
            save_pkl=save_pkl, save_fasta=save_fasta, save_csv=save_csv)

    if save_lib:
        lib_dir = save_dir / (args.auto_prefix if save_name is None else save_name)
        save_individual_files(df, save_dir=lib_dir, save_header=None,
            save_genre=save_genre, named_after=named_after,
            tqdm_disable=tqdm_disable)


def bake_rfam_ss_consensus(
        data_name,
        save_dir='./',
        PPavg_topcut=0.7,
        PPavg_cutoff=0.5,
        save_individual=False,
        save_lumpsum=True,
        **kwargs):
    """ bake rfam ss_consensus structure files into midat """
    args = misc.Struct(locals())
    logger.debug('Arguments:\n' + gwio.json_str(args.__dict__))

    save_dir = Path(save_dir) / Path(data_name).stem
    rfam_align = molalign.AlignRNAStockholm(data_name)
    rfam_align.save_individual_fasta_dbns(PPavg_cutoff=PPavg_cutoff, PPavg_topcut=PPavg_topcut,
            save_dir=save_dir, save_individual=save_individual, save_lumpsum=save_lumpsum)



def bake_eternabench_chemmap(
        data_names,
        primer5_seq=None,
        primer3_seq='AAAAGAAACAACAACAACAAC',
        padding=-1.0,        # value for missing chemical activities
        save_trimmed=False,  # save with seq_trimmed as seq (for cd-hit-est run)
        save_prefix=None,
        tqdm_disable=False,
        **kwargs):
    """ convert EternaBench dataset into dataframe (see comments for further details) """
    args = misc.Struct(locals())
    logger.debug('Arguments:\n'+gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix.replace('.json', ''), 'autosave'])

    # reform to my conventions...
    column_mappings = {
        "ID": 'id',
        "Dataset": "dataset",
        'sequence': 'seq',
        'reactivity': 'chemical_activity',
        'errors': 'chemical_activity_err',
    }
    df.rename(columns=column_mappings, inplace=True)

    # Notes on chemical activity data. 
    # It appears to involve the following steps:
    #   1) some chemical (can be just water) is used to degrade an ensemble of RNA molecules
    #      One example is Selective 2′ Hydroxyl Acylation analyzed by Primer Extension (SHAPE)
    #      where 1M7 compound is used to modify 2'-OH group and terminate primer extension.
    #   2) the incubation time is chosen such that the modification probability of any base
    #      is small, aiming to achieve so-called "single-hit kinetics" where the probability 
    #      is linearly proportional to the rate constant for each structural state. Then, in a
    #      simplest two-structure-state model (paired vs. unpaired), the modification probability
    #      is linear with respect to the pairing probability.
    #   3) Chemical mapping used MAP-seq protocol. CD-HIT-EST was used to filter sequences with
    #      >80% redundancy by keeping the one with highest signal to noise in each cluster.
    #   4) reactivities less than or equal to zero or greater than 95% of all nucleotides are
    #      removed from analysis. 
    #      Note that the binding factor for Riboswitch is added for some
    #      studies (e.g., with FMN) and should be removed. Adenosine nucleotides preceded by six
    #      or more As should also be removed due to evidence of anomalous reverse transcription
    #      effects. 
    #   5) considerations in 4) is also applicable to this type of data beyond EternaBench studies.

    # add columns 
    df['idx'] = list(range(len(df)))
    df['len'] = df['seq'].str.len()
    # df['chemical_activity_had_zeros'] = False
    # df['chemical_activity_val_counts'] = 0
    df['seqpos_overflow'] = [np.empty((0,), dtype=np.int32)] * len(df)
    df['chemical_activity_overflow'] = [np.empty((0,), dtype=np.float32)] * len(df)
    df['chemical_activity_err_overflow'] = [np.empty((0,), dtype=np.float32)] * len(df)
    
    if 'signal_to_noise' in df:
        signal_to_noise = df['signal_to_noise'].str.split(':', expand=True)
        signal_to_noise.set_axis(['signal_to_noise_level', 'signal_to_noise_ratio'], axis=1, inplace=True)
        df = pd.concat([df, signal_to_noise], axis=1, copy=False)
        type_mappings = {'id': str, 'signal_to_noise_ratio': np.float32}
    else:
        type_mappings = {'id': str}

    if 'EternaScore' in df: type_mappings['EternaScore'] = np.float32
    df = df.astype(type_mappings)
        
    for col in ['seqpos']:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.int32))
        
    for col in ['chemical_activity', 'chemical_activity_err']:
        if col not in df: continue
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float32))
    
    # get trimmed sequences by removing 5/3' primers
    if primer5_seq is not None or primer3_seq is not None:
        if primer5_seq is not None:
            outlier_ids = df.loc[~df['seq'].str.startswith(primer5_seq), 'id']
            if len(outlier_ids):
                logger.warning(f'Not all sequences start with primer5_seq: {primer5_seq}! # of outliers: {len(outlier_ids)}')
                print(outlier_ids)
        if primer3_seq is not None:
            outlier_ids = df.loc[~df['seq'].str.endswith(primer3_seq), 'id']
            if len(outlier_ids):
                logger.warning(f'Not all sequences end with primer3_seq: {primer3_seq}!  # of outliers: {len(outlier_ids)}')
                print(outlier_ids)                

        istart = 0 if primer5_seq is None else len(primer5_seq)
        df['seq_trimmed'] = df['seq'].str[istart:] if primer3_seq is None else df['seq'].str[istart:-len(primer3_seq)]
        df['len_trimmed'] = df['seq_trimmed'].str.len()
    else:
        df['seq_trimmed'] = df['seq']
        df['len_trimmed'] = df['len']

    # check 'seqpos' and pad missing chemical reactivity values
    good2go_list, conflict_list = [], []
    icol_len = df.columns.get_loc('len')
    icol_seqpos = df.columns.get_loc('seqpos')
    icol_chemact = df.columns.get_loc('chemical_activity')
    icol_chemact_err = df.columns.get_loc('chemical_activity_err')


    icol_seqpos_overflow = df.columns.get_loc('seqpos_overflow')
    icol_chemact_overflow = df.columns.get_loc('chemical_activity_overflow')
    icol_chemact_err_overflow = df.columns.get_loc('chemical_activity_err_overflow')

    # iat_chemact_zeros = df.columns.get_loc('chemical_activity_had_zeros')
    # iat_chemact_counts = df.columns.get_loc('chemical_activity_val_counts')
    for i in tqdm(range(len(df)), total=len(df)): # df.iterrows() creates a copy of the row
        ds = df.iloc[i]

        # 1) check inconsistency between seqpos and chemical_activity (none was ever found)
        if len(df.iat[i, icol_chemact]) != len(df.iat[i, icol_seqpos]):
            logger.warning('Different chemical_activity and seqpos lengths!')
            print(ds)
            conflict_list.append(i)
            continue

        # 2) check for duplicated seqpos (none was ever found)
        seqpos_duplicates = np.sort(df.iat[i, icol_seqpos], axis=0)
        seqpos_duplicates = seqpos_duplicates[np.where(np.diff(seqpos_duplicates, axis=0) == 0)[0]]

        if len(seqpos_duplicates):
            logger.warning('Duplicated seqpos exists!')
            print(ds)
            conflict_list.append(i)
            continue

        good2go_list.append(i)

        # 3) move away the seqpos and chemical_activity values exceeding the sequence length
        # These chemical_activity values however turn out to be zero or negative (later ignored anyway). 

        # The obsolete code below removes all zero and negative values (decided to keep them for later...)
        '''
        idx_nonzeros = np.where(df.iat[i, iat_chemact] > 0.0)[0]
        if len(idx_nonzeros) != len(ds.chemical_activity) :
            df.iat[i, iat_chemact_zeros] = True
            df.iat[i, iat_seqpos] = df.iat[i, iat_seqpos][idx_nonzeros]
            df.iat[i, iat_chemact] = df.iat[i, iat_chemact][idx_nonzeros]
            df.iat[i, iat_chemact_err] = df.iat[i, iat_chemact_err][idx_nonzeros]
        df.iat[i, iat_chemact_counts] = len(idx_nonzeros)
        
        if len(df.iat[i, iat_seqpos]) and (df.iat[i, iat_seqpos][-1] >= df.iat[i, iat_len] or len(df.iat[i, iat_seqpos]) > df.iat[i, iat_len]):
            logger.warning(f'Greater or longer seqpos: {len(df.iat[i, iat_seqpos])} than sequence length: {ds.iat[i, iat_len]}!')
            print(ds)
            conflict_list.append(i)
            continue
        '''
        idx_overflow = np.where(df.iat[i, icol_seqpos] >= df.iat[i, icol_len])[0]
        if len(idx_overflow):
            df.iat[i, icol_seqpos_overflow] = df.iat[i, icol_seqpos][idx_overflow]
            df.iat[i, icol_seqpos] = df.iat[i, icol_seqpos][:idx_overflow[0]]
            df.iat[i, icol_chemact_overflow] = df.iat[i, icol_chemact][idx_overflow]
            df.iat[i, icol_chemact] = df.iat[i, icol_chemact][:idx_overflow[0]]
            df.iat[i, icol_chemact_err_overflow] = df.iat[i, icol_chemact_err][idx_overflow]
            df.iat[i, icol_chemact_err] = df.iat[i, icol_chemact_err][:idx_overflow[0]]

        # 4) pad the missing chemical activities (most entries actually miss)
        chemical_activity = np.full((df.iat[i, icol_len],), fill_value=padding, dtype=np.float32)
        chemical_activity[df.iat[i, icol_seqpos]] = df.iat[i, icol_chemact]
        df.iat[i, icol_chemact] = chemical_activity

        chemical_activity_err = np.full((df.iat[i, icol_len],), fill_value=padding, dtype=np.float32)
        chemical_activity_err[df.iat[i, icol_seqpos]] = df.iat[i, icol_chemact_err]
        df.iat[i, icol_chemact_err] = chemical_activity_err

    df['len_seqpos'] = df.seqpos.apply(len)
    df['len_seqpos_overflow'] = df.seqpos_overflow.apply(len)
    df.fillna(padding, inplace=True)

    # df.drop(columns=['seqpos', 'signal_to_noise'], inplace=True)

    kwargs.setdefault('save_pkl', True)
    kwargs.setdefault('save_lumpsum', True)
    if len(good2go_list):
        save_files(df.iloc[good2go_list], save_prefix=save_prefix, **kwargs)
        if save_trimmed:
            save_files(df.iloc[good2go_list], save_prefix=f'{save_prefix}_trimmed',
                id_key='id', fasta_key='seq_trimmed', **kwargs)

    if len(conflict_list):
        save_files(df.iloc[conflict_list], save_prefix=f'{save_prefix}_conflict', **kwargs)


def bake_eternabench_external(
        data_names,
        padding=-1.0,        # value for missing chemical activities
        save_trimmed=False,  # save with seq_trimmed as seq (for cd-hit-est run)
        save_prefix=None,
        tqdm_disable=False,
        **kwargs):
    """ convert EternaBench external dataset into dataframe (see comments for further details) """
    args = misc.Struct(locals())
    logger.debug('Arguments:\n'+gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix.replace('.json', ''), 'autosave'])

    # consider to use "name" as "id"
    if "ID" not in df:
        df["ID"] = list(range(len(df)))
    
    # reform to my conventions...
    column_mappings = {
        "ID": 'id',
        "Dataset": "dataset",
        "Class": 'class',
        'sequence': 'seq',
        'reactivity': 'chemical_activity',
    }
    df.rename(columns=column_mappings, inplace=True)

    # add columns 
    df['idx'] = list(range(len(df)))
    df['len'] = df['seq'].str.len()
    # df['chemical_activity_had_zeros'] = False
    # df['chemical_activity_val_counts'] = 0
    df['seqpos_overflow'] = [np.empty((0,), dtype=np.int32)] * len(df)
    df['chemical_activity_overflow'] = [np.empty((0,), dtype=np.float32)] * len(df)

    type_mappings = {'id': str}
    df = df.astype(type_mappings)
        
    for col in ['seqpos']:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.int32))
        
    for col in ['chemical_activity', 'chemical_activity_err']:
        if col not in df: continue
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float32))

    # check 'seqpos' and pad missing chemical reactivity values
    good2go_list, conflict_list = [], []
    icol_len = df.columns.get_loc('len')
    icol_seqpos = df.columns.get_loc('seqpos')
    icol_chemact = df.columns.get_loc('chemical_activity')
    icol_seqpos_overflow = df.columns.get_loc('seqpos_overflow')
    icol_chemact_overflow = df.columns.get_loc('chemical_activity_overflow')

    for i in tqdm(range(len(df)), total=len(df)): # df.iterrows() creates a copy of the row
        ds = df.iloc[i]

        # 1) check inconsistency between seqpos and chemical_activity (none was ever found)
        if len(df.iat[i, icol_chemact]) != len(df.iat[i, icol_seqpos]):
            logger.warning('Different chemical_activity and seqpos lengths!')
            print(ds)
            conflict_list.append(i)
            continue

        # 2) check for duplicated seqpos (none was ever found)
        seqpos_duplicates = np.sort(df.iat[i, icol_seqpos], axis=0)
        seqpos_duplicates = seqpos_duplicates[np.where(np.diff(seqpos_duplicates, axis=0) == 0)[0]]

        if len(seqpos_duplicates):
            logger.warning('Duplicated seqpos exists!')
            print(ds)
            conflict_list.append(i)
            continue

        good2go_list.append(i)

        # 3) move away the seqpos and chemical_activity values exceeding the sequence length
        # These chemical_activity values however turn out to be zero or negative (later ignored anyway). 

        # The obsolete code below removes all zero and negative values (decided to keep them for later...)
        '''
        idx_nonzeros = np.where(df.iat[i, iat_chemact] > 0.0)[0]
        if len(idx_nonzeros) != len(ds.chemical_activity) :
            df.iat[i, iat_chemact_zeros] = True
            df.iat[i, iat_seqpos] = df.iat[i, iat_seqpos][idx_nonzeros]
            df.iat[i, iat_chemact] = df.iat[i, iat_chemact][idx_nonzeros]
            df.iat[i, iat_chemact_err] = df.iat[i, iat_chemact_err][idx_nonzeros]
        df.iat[i, iat_chemact_counts] = len(idx_nonzeros)
        
        if len(df.iat[i, iat_seqpos]) and (df.iat[i, iat_seqpos][-1] >= df.iat[i, iat_len] or len(df.iat[i, iat_seqpos]) > df.iat[i, iat_len]):
            logger.warning(f'Greater or longer seqpos: {len(df.iat[i, iat_seqpos])} than sequence length: {ds.iat[i, iat_len]}!')
            print(ds)
            conflict_list.append(i)
            continue
        '''
        idx_overflow = np.where(df.iat[i, icol_seqpos] >= df.iat[i, icol_len])[0]
        if len(idx_overflow):
            df.iat[i, icol_seqpos_overflow] = df.iat[i, icol_seqpos][idx_overflow]
            df.iat[i, icol_seqpos] = df.iat[i, icol_seqpos][:idx_overflow[0]]
            df.iat[i, icol_chemact_overflow] = df.iat[i, icol_chemact][idx_overflow]
            df.iat[i, icol_chemact] = df.iat[i, icol_chemact][:idx_overflow[0]]

        # 4) pad the missing chemical activities (most entries actually miss)
        chemical_activity = np.full((df.iat[i, icol_len],), fill_value=padding, dtype=np.float32)
        chemical_activity[df.iat[i, icol_seqpos]] = df.iat[i, icol_chemact]
        df.iat[i, icol_chemact] = chemical_activity

    df['len_seqpos'] = df.seqpos.apply(len)
    df['len_seqpos_overflow'] = df.seqpos_overflow.apply(len)
    df.fillna(padding, inplace=True)

    # df.drop(columns=['seqpos', 'signal_to_noise'], inplace=True)

    kwargs.setdefault('save_pkl', True)
    kwargs.setdefault('save_lumpsum', True)
    if len(good2go_list):
        save_files(df.iloc[good2go_list], save_prefix=save_prefix, **kwargs)
        if save_trimmed:
            save_files(df.iloc[good2go_list], save_prefix=f'{save_prefix}_trimmed',
                id_key='id', fasta_key='seq_trimmed', **kwargs)

    if len(conflict_list):
        save_files(df.iloc[conflict_list], save_prefix=f'{save_prefix}_conflict', **kwargs)


RNA_SS_data = namedtuple('RNA_SS_data','seq ss_label length name pairs', defaults=(None,) * 5)
def clone_midat2ufold(data_names,         # dataframe in csv, pkl
                      save_prefix=None,   #
                      tqdm_disable=False, # NOTE: see save_all_files for saving args!!!
                      **kwargs):
    """ convert homebrew dataframe to ufold namedtuples """
    args = misc.Struct(locals())
    logger.debug('Arguments:\n'+gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'autosave'])


    ufold_tuple_list = []
    for idx, ds in tqdm(df.iterrows(), total=len(df), disable=tqdm_disable):
        ufold_tuple_list.append(RNA_SS_data(
            seq = molstru.seq2onehot(molstru.seq2RNA(ds.seq), length=600),
            ss_label = molstru.dbn2onehot(molstru.ct2dbn(ds.ct, len(ds.seq)), length=600),
            length = len(ds.seq),
            name = ds['id'],
            pairs = (ds.ct - 1) if len(ds.ct) else [],
        ))

    save_file = f"{save_prefix}.cPickle"
    logger.info(f'Saving UFold file: {save_file} ...')
    with open(save_file, 'wb') as iofile:
        pickle.dump(ufold_tuple_list, iofile)


def clone_seq2rna(data_names,           # dataframe in csv, pkl
                  tqdm_disable=False,   # NOTE: see save_all_files for saving args!!!
                  **kwargs):
    """ convert all nucleotides to AUCG """

    args = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if auto_save_prefix is None: auto_save_prefix = 'autosave'

    if len(df) > 42:
        df['seq'] = misc.mpi_map(molstru.seq2RNA, tqdm(df['seq'], disable=tqdm_disable))
    else:
        df['seq'] = [molstru.seq2RNA(seq) for seq in df['seq']]

    save_files(df, tqdm_disable=tqdm_disable, **kwargs)


def clone_seq2kmer(data_names,           # dataframe in csv, pkl
                   save_dir='./',        # current dir is default
                   save_prefix=None,     # default names will be used if not set
                   k=5, padding='N',     # seq2kmer parameters
                   test_size=0.11,       # percentage of test dataset
                   num_cpus=0.6,         # number/percentage of CPUs
                   tqdm_disable=False,   # disable tqdm
                   **kwargs):
    """ convert each seq into its kmer representations """

    save_dir = Path(save_dir)
    args = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    args.auto_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'noname'])

    fn_seq2kmer = functools.partial(molstru.seq2kmer, k=k, padding=padding)

    args.num_seqs = len(df)
    logger.info(f'Kmerize {args.num_seqs} sequences...')
    kmer_seqs = misc.mpi_map(fn_seq2kmer,
            df.seq if tqdm_disable else tqdm(df.seq, desc='Seq2Kmer'),
            num_cpus=num_cpus, quiet=True)

    assert len(kmer_seqs) == len(df), 'Length changed after seq2kmer'

    bert_label = f'_bert{k}' + padding if padding else ''
    args.save_path = save_dir / (args.auto_prefix + bert_label)
    gwio.dict2json(args.__dict__, fname=args.save_path.with_suffix('.json'))

    # split the dataset and save
    train_seqs, test_seqs = train_test_split(kmer_seqs, test_size=test_size)

    save_path = save_dir / (args.auto_prefix + bert_label + '_train.txt')
    logger.info(f'Saving BERT train sequences to txt file: {save_path}')
    with save_path.open('w') as iofile:
        for seq in train_seqs if tqdm_disable else tqdm(train_seqs, desc='Save train'):
            iofile.writelines([seq, '\n'])

    save_path = save_dir / (args.auto_prefix + bert_label + '_test.txt')
    logger.info(f'Saving BERT test sequences to txt file: {save_path}')
    with save_path.open('w') as iofile:
        for seq in tqdm(test_seqs, desc='Save test'):
            iofile.writelines([seq, '\n'])

    # save to fasta
    # save_path = save_dir / (save_prefix + bert_label + '.fasta')
    # logger.info(f'Saving BERT sequences to fasta file: {save_path}')
    # with save_path.open('w') as iofile:
    #     for i, seq in tqdm(enumerate(kmer_seqs)):
    #         iofile.writelines([f'>BERT-SEQ{i + 1}\n',
    #             molstru.kmer2seq(seq, padding=padding), '\n'])

    # save_path = save_dir / (save_prefix + bert_label + '.txt')
    # logger.info(f'Saving BERT sequences to txt file: {save_path}')
    # with save_path.open('w') as iofile:
    #     for seq in tqdm(kmer_seqs):
    #         iofile.writelines([seq, '\n'])
    return None


def clone_kmer2seq(
        data_names,                     # kmer sequence file (line by line)
        save_name=None, save_dir='./',  # default: noname
        padding=None,                   # whether to use
        tqdm_disable=False,
        **kwargs):
    """ infer seqence from its bert kmer sentence """

    data_names = misc.unpack_list_tuple(data_names)
    bert_seqs = gwio.get_file_lines(data_names, strip=True, keep_empty=False)
    save_name = misc.get_1st_value([save_name, 'noname'])
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # show parameters?
    logger.info(f'   save_name: {save_name}')
    logger.info(f'    save_dir: {save_dir}')
    logger.info(f'     padding: {padding}')
    logger.info(f'   # of seqs: {len(bert_seqs)}')

    #
    fasta_seqs = []
    for seq in bert_seqs:
        fasta_seqs.append(molstru.kmer2seq(seq, padding=padding))

    save_path = (save_dir / save_name).with_suffix('.fasta')
    logger.info(f'Saving fasta file: {save_path}...')
    id_base = Path(data_names[0]).stem
    with save_path.open('w') as iofile:
        for i, seq in enumerate(fasta_seqs):
            iofile.writelines([f'>{id_base}_fasta{i}\n', fasta_seqs[i], '\n'])


def compare_dataset_pairs(
        data_names,
        fmt='pkl',
        key='seq',
        val='ct',
        save_dir='./',
        save_name=None,
        **kwargs):
    """ compare two datasets for intersection, union, etc. """
    if len(data_names) > 2:
        logger.warning('Only the first two files will be processed!')

    save_dir = Path(save_dir)

    # read files
    df_list, df_file = gwio.get_dataframes(data_names[0:2], fmt=fmt, concat=False, return_files=True)

    len_df = [_df.shape[0] for _df in df_list]
    df1 = df_list[0].assign(pkl_src=1, keep_me=True)
    df2 = df_list[1].assign(pkl_src=2, keep_me=True)
    df_all = pd.concat((df1, df2), axis=0, ignore_index=True, copy=False)

    # intersection (assume each df is already unique)
    df_intersect = df_all.groupby(by=key).filter(lambda grp: len(grp) > 1)

    df_grps = df_intersect.groupby(by=key)
    df_intersect_unique = df_grps.head(1)
    # check consistency in the intersection
    for seq, df_grp in df_grps:

        val0 = df_grp.iloc[0][val]
        same_grp_vals = [True]
        for igrp in range(1, len(df_grp)):
            same_grp_vals.append(np.array_equal(val0, df_grp.iloc[igrp][val]))

        if all(same_grp_vals): # consistent vals, keep the first one
            df_all.loc[df_grp.index[1:], 'keep_me'] = False
            # df_intersect.loc[df_grp.index[1:], 'saved'] = False
            logger.info(f'The same {key} and {val} for the following:')
            print(df_grp['id'])
        else: # inconsistent vals, keep no one
            df_all.loc[df_grp.index, 'keep_me'] = False
            df_intersect.loc[df_grp.index, 'keep_me'] = False
            logger.warning(f'The same {key} but different {val} for the following:')
            print(df_grp['id'])

    df_intersect_same = df_intersect[df_intersect.keep_me == True]
    df_intersect_diff = df_intersect[df_intersect.keep_me == False]

    # union
    df_union = df_all[df_all['keep_me'] == True]

    # save the unique ones for each list
    df1_unique = df1[df1[key].isin(df_intersect_unique[key]) == False]
    df2_unique = df2[df2[key].isin(df_intersect_unique[key]) == False]

    logger.info(f'DataFrame #1 shape: {df1.shape}')
    logger.info(f'DataFrame #1 shape: {df2.shape}')
    logger.info(f'Union shape: {df_union.shape}')
    logger.info(f'Intersect shape: {df_intersect.shape}')
    logger.info(f'Intersect same shape: {df_intersect_same.shape}')
    logger.info(f'Intersect diff shape: {df_intersect_diff.shape}')
    logger.info(f'Intersect unique shape: {df_intersect_unique.shape}')
    logger.info(f'DataFrame #1 Unique shape: {df1_unique.shape}')
    logger.info(f'DataFrame #2 Unique shape: {df2_unique.shape}')

    save_file = save_dir / 'intersect.pkl'
    logger.info(f'Saving pickle file: {save_file}')
    df_intersect.to_pickle(save_file, protocol=-1, compression='infer')


def gather_seq_pdb(data_names=None,
                   data_dir='./',
                   seq_suffix='.fasta',
                   ct_suffix='.ct',
                   pdb_suffix='.pdb',
                   angle_suffix='.json',
                   split_data=True, split_size=0.1,
                   min_len=1,
                   max_len=3200000000,
                   save_dir='./',
                   save_pkl=True,
                   save_lib=True,
                   tqdm_disable=False,
                   debug=False,
                   **kwargs):
    """ build dataset from seq and pdb databases """
    recap = locals()
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # get the sequence file
    seq_files = list(data_dir.glob('**/*' + seq_suffix))
    logger.info(f'Searched dir: {data_dir}, found {len(seq_files)} seq files')

    if debug:
        seq_files = seq_files[:77]

    # split the files
    all_files = dict()
    if split_data:
        data_groups = ['train', 'valid']
        all_files['train'], all_files['valid'] = train_test_split(seq_files, test_size=split_size,
                        shuffle=True, random_state=20130509)
    else:
        data_groups = ['train']
        all_files['train'] = seq_files

    # just do it
    for data_group in data_groups:
        logger.info(f'Processing data type: {data_group}...')

        seq_data = molstru.SeqStruct2D(all_files[data_group], fmt=seq_suffix[1:])
        num_seqs = len(seq_data.seq)
        logger.info(f'Successfully parsed {num_seqs} seq files')

        pdb_data = misc.Struct(num=np.zeros(num_seqs, dtype=int),
                               len=np.zeros(num_seqs, dtype=int),
                               seq=[''] * num_seqs,
                               pdist=[None] * num_seqs,
                               valid=np.full(num_seqs, False) ,
                               )

        ct_data = misc.Struct(num=np.zeros(num_seqs, dtype=int),
                              len=np.zeros(num_seqs, dtype=int),
                              seq=[''] * num_seqs,
                              id=[''] * num_seqs,
                              ct=[np.empty(0, dtype=int)] * num_seqs,
                              valid=np.full(num_seqs, False) ,
                              )

        angle_data = misc.Struct(num=np.zeros(num_seqs, dtype=int),
                                len=np.zeros(num_seqs, dtype=int),
                                torsion=[None] * num_seqs,
                                valid=np.full(num_seqs, False),
                                )

        seq_data.valid = np.logical_and(seq_data.len >= min_len, seq_data.len <= max_len)

        for iseq, seq_file in enumerate(seq_data.file):

            if not seq_data.valid[iseq]:
                continue

            # pdb file
            _pdb_file = Path(seq_file).with_suffix(pdb_suffix)
            if _pdb_file.exists():
                pdb_data.num[iseq] = 1

                _pdb_data = molstru.AtomsData(_pdb_file.as_posix())
                pdb_data.len[iseq] = _pdb_data.numResids

                _pdb_seq = ''.join(_pdb_data.seqResidNames)
                if seq_data.len[iseq] == pdb_data.len[iseq]:
                    if seq_data.seq[iseq] == _pdb_seq: # a perfect match!
                        pdb_data.valid[iseq] = True
                    else:
                        logger.warning(f'[{seq_file}]: pdb has the same len but different seq!')
                else:
                    logger.warning(f'[{seq_file}]: pdb has different length and sequence!')

                if not pdb_data.valid[iseq]:
                    pdb_data.seq[iseq] = _pdb_seq
                    print(f'[{_pdb_file}]: SEQ: {seq_data.len[iseq]} and PDB: {_pdb_data.numResids}')
                    print(f'Sequence mismatch for {_pdb_file} -->')
                    print(f'SEQ: {seq_data.seq[iseq]}')
                    print(f'PDB: {pdb_data.seq[iseq]}')

                # calculate the distance matrix and save
                _pdb_data.calc_res2res_distance(atom='P', neighbor_only=False)
                pdb_data.pdist[iseq] = _pdb_data.dist_res2res_byatom

            # obtain&check ct file
            _ct_file = Path(seq_file).with_suffix(ct_suffix)
            if _ct_file.exists():
                ct_data.num[iseq] = 1
            else:
                _ct_file = list(_ct_file.parent.glob(_ct_file.stem + '_p*' + ct_suffix))
                ct_data.num[iseq] = len(_ct_file)
                if len(_ct_file):
                    logger.info(f'{len(_ct_file)} ct files found for {seq_file}')
                    print(_ct_file)
                    _ct_file = _ct_file[0]

            if ct_data.num[iseq] > 0: # process ct file
                _ct_data = molstru.parse_ct_lines(_ct_file, is_file=True)
                ct_data.id[iseq] = _ct_data['id']
                ct_data.len[iseq] = _ct_data['len']
                ct_data.ct[iseq] = _ct_data['ct']

                if seq_data.len[iseq] == ct_data.len[iseq]:
                    if seq_data.seq[iseq] == _ct_data['seq']: # a perfect match!
                        ct_data.valid[iseq] = True
                    else:
                        logger.warning(f'[{seq_file}]: ct has the same len but different seq!')

                elif seq_data.len[iseq] > _ct_data['len']:
                    # check whether a subset of seqdata.seq[iseq]
                    _seq_seq = ''.join([seq_data.seq[iseq][_i - 1] for _i in _ct_data['resnum']])
                    _ct_seq = ''.join([_ct_data['seq'][_i - 1] for _i in _ct_data['resnum']])
                    if _seq_seq == _ct_seq:
                        ct_data.valid[iseq] = True
                    else:
                        logger.warning(f'[{seq_file}]: ct has different length and sequence!')

                if not ct_data.valid[iseq]:
                    ct_data.seq[iseq] = _ct_data['seq']
                    print(f'[{_ct_file}]: SEQ: {seq_data.len[iseq]} and CT: {_ct_data["len"]}')
                    print(f'Sequence mismatch for {_ct_file} -->')
                    print(f'SEQ: {seq_data.seq[iseq]}')
                    print(f' CT: {ct_data.seq[iseq]}')

            # angle file
            _angle_file = Path(seq_file).with_suffix(angle_suffix)
            if _angle_file.exists():

                with _angle_file.open('r') as iofile:
                    try: # sometimes the json file is corrupted
                        _angle_data = json.load(iofile)
                    except:
                        _angle_data = dict()

                angle_data.num[iseq] = 1

                # get the torsion angles form json file
                if 'nts' in _angle_data and seq_data.len[iseq] == len(_angle_data['nts']):
                    angle_data.valid[iseq] = True
                    angle_data.len[iseq] = len(_angle_data['nts'])
                    angle_data.torsion[iseq] = [[_nt['alpha'], _nt['beta'], _nt['gamma'],
                                                 _nt['delta'], _nt['epsilon'], _nt['zeta']]
                                                 for _nt in _angle_data['nts']]
                    # angle_data.torsion[iseq][0][:2] = 0.0, 0.0 # alpha and beta for 1st nt
                    # angle_data.torsion[iseq][-1][-2:] = 0.0, 0.0 # epsilon and zeta for last nt
                    # there are more None values though!
                    angle_data.torsion[iseq] = np.stack(angle_data.torsion[iseq], axis=0)
                    angle_data.torsion[iseq][np.where(angle_data.torsion[iseq] == None)] = 0.0
                else:
                    logger.warning(f'[{seq_file}]: angle has different seq length!')

        # determine whether to save each seq
        seq2sav = np.logical_and(ct_data.valid, pdb_data.valid)
        seq2sav = np.logical_and(seq2sav, angle_data.valid)

        # save csv
        df = seq_data.to_df()
        df = df.assign(saved=seq2sav, validCT=ct_data.valid, numCT=ct_data.num, lenCT=ct_data.len,
                       lenDiffCT=seq_data.len - ct_data.len, seqCT = ct_data.seq,
                       validPDB=pdb_data.valid, numPDB=pdb_data.num, lenPDB=pdb_data.len,
                       lenDiffPDB=seq_data.len - pdb_data.len, seqPDB=pdb_data.seq,
                       validAngle=angle_data.valid, numAngle=angle_data.num,
                       lenAngle=angle_data.len,
                       )

        csv_file = save_dir / (data_group + '.csv')
        logger.info(f'Storing csv: {csv_file}')
        df.to_csv(csv_file, index=False, float_format='%8.6f')

        # remove invalid sequences
        logger.info(f'{seq_data.valid.astype(int).sum()} out of {len(seq_data.seq)} are within' + \
                    f' length range of [{min_len}, {max_len}]')
        idx2sav = np.where(seq2sav)[0]
        logger.info(f'{len(idx2sav)} entries have valid ct, pdb, and angle data')
        seq_data = seq_data.get_subset(idx2sav)

        if save_pkl:
            midata = dict(id=seq_data.id, len=seq_data.len, seq=seq_data.seq,
                          ct=[ct_data.ct[_i] for _i in idx2sav],
                          pdist=[pdb_data.pdist[_i] for _i in idx2sav],
                          angle=[angle_data.torsion[_i] for _i in idx2sav],
            )

            pkl_file = save_dir / (data_group + '.pkl')
            logger.info(f'Storing pickle: {pkl_file}')
            gwio.pickle_squeeze(midata, pkl_file, fmt='lzma')

        if save_lib:
            lib_dir = save_dir / data_group
            lib_dir.mkdir(exist_ok=True)
            logger.info(f'Saving lib files in: {lib_dir}')

            with tqdm(total=len(seq_data.seq), disable=tqdm_disable) as prog_bar:
                for i in range(len(seq_data.seq)):
                    sav_file = lib_dir / f'{i + 1:d}.input.fasta'
                    with sav_file.open('w') as iofile:
                        iofile.writelines('\n'.join(
                            seq_data.get_fasta_lines(i, dbn=False, upp=False)
                        ))

                    if data_group != 'predict':
                        ct_mat = molstru.ct2ctmat(ct_data.ct[idx2sav[i]], seq_data.len[i])
                        sav_file = lib_dir / f'{i + 1:d}.label.ctmat'
                        np.savetxt(sav_file, ct_mat, fmt='%1i')

                        sav_file = lib_dir / f'{i + 1:d}.label.pdist'
                        np.savetxt(sav_file, pdb_data.pdist[idx2sav[i]], fmt='%8.2f')

                        sav_file = lib_dir / f'{i + 1:d}.label.angle'
                        np.savetxt(sav_file, angle_data.torsion[idx2sav[i]], fmt='%8.2f')

                    if (i + 1) % 10 == 0:
                        prog_bar.update(10)
                prog_bar.update((i + 1) % 10)


def add_res2res_mat2d(data_names,
        data_dir=None,          # folder for the dist files
        suffix='dist',
        key=None,               # the key/column name to be added (default: suffix)
        matchby='srcfile',      # change to alternative cols such as "id" as appropriate
        max_len=None,
        save_prefix=None,
        save_pkl=True,
        **kwargs):
    """ add a single mat2d to each row as a new column """
    args = misc.Struct(locals())
    logger.info('Arguments:\n' + gwio.json_str(args.__dict__))

    # load base midata
    key = misc.get_1st_value([key, suffix])
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value((save_prefix, f'{auto_save_prefix}_{key}', 'noname'))

    if max_len is not None:
        logger.info(f'Removing sequences longer than {max_len}, original number: {len(df)}')
        df['len'] = df.seq.str.len()
        df = df[df['len'] <= max_len]
        logger.info(f'The number of sequences left: {len(df)}')

    data_dir = misc.get_1st_value([data_dir, './'])

    mat2d_files = [os.path.join(data_dir, f'{Path(molid).stem}.{suffix}') for molid in df[matchby]]

    mat2d_data = misc.mpi_map(molstru.parse_res2res_mat2d_lines, mat2d_files, tqdm_disable=False)
    df[key] = [None if _mat2d is None else _mat2d['mat2d']  for _mat2d in mat2d_data]

    if save_pkl:
        save_file = f'{save_prefix}.pkl'
        logger.info(f'Saving pickle file: {save_file} ...')
        df.to_pickle(save_file)
    return df


def add_res2res_mat2ds(data_names,
        data_dir=None,                      # folder for the dist files
        suffixes=['dist', 'PCCP', 'CNNC'],  # any res2res mat2d should work in princple
        key='disdihedral',                  # all angles will be combined into one key
        matchby='srcfile',                  # change to alternative cols such as "id" as appropriate
        save_prefix=None,                   # default to {data_names.stem}_{key}
        save_pkl=True,
        **kwargs):
    """ stack and add multiple mat2ds to each row as a new column """
    args = misc.Struct(locals())
    logger.info('Arguments:\n' + gwio.json_str(args.__dict__))

    # load base midata
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value((save_prefix, f'{auto_save_prefix}_{key}', 'noname'))

    data_dir = misc.get_1st_value([data_dir, './'])

    mat2ds_data = []
    nones_isa = [False] * len(df)
    for suffix in suffixes:
        mat2d_files = [os.path.join(data_dir, f'{Path(molid).stem}.{suffix}') for molid in df[matchby]]
        mat2d_data = misc.mpi_map(molstru.parse_res2res_mat2d_lines, mat2d_files, tqdm_disable=False)
        mat2ds_data.append([None if _mat2d is None else _mat2d['mat2d']  for _mat2d in mat2d_data])
        nones_isa = [True if _mat2d is None else nones_isa[_i] for _i, _mat2d in enumerate(mat2d_data)]

    idx2remove = []
    for i, none_isa in enumerate(nones_isa):
        if none_isa:
            logger.warning(f'Missing dihedral data for idx: {i}, id: {df.iloc[i][matchby]} - to be removed!')
            idx2remove.append(i)

    if len(idx2remove):
        logger.info(f'Dropping {len(idx2remove)} rows from dataframe ...')
        df.drop(index=df.index[idx2remove], inplace=True)
        logger.info(f'Current dataframe shape: {df.shape}')
        
        mat2ds_data = [[_mat2d for _i, _mat2d in enumerate(mat2d_data) if not nones_isa[_i]] for mat2d_data in mat2ds_data]
        
    if len(suffixes) > 1:
        mat2ds_data = list(zip(*mat2ds_data))
        mat2ds_data = [np.stack(_mat2ds, axis=-1) for _mat2ds in mat2ds_data]
    else:
        mat2ds_data = mat2ds_data[0]

    df[key] = mat2ds_data

    if save_pkl:
        save_file = f'{save_prefix}.pkl'
        logger.info(f'Saving pickle file: {save_file} ...')
        df.to_pickle(save_file)
    return df    


def gather_fasta_st(**kwargs):
    """ a wrapper for gather_seq_ct with seq_fmt as fasta """
    args = dict(seq_fmt='fasta', seq_suffix='.fasta', ct_fmt='st', ct_suffix='.st')
    args.update(kwargs)
    gather_seq_ct(**args)


def gather_seq_ct(
        file_prefixes=None, # prefixes are passed, usually just one file (not implemented yet!)
        data_base=None,     # database: archiveii, stralign, bprna, etc.
        data_dir='./',
        seq_dir=None,       # default to data_dir
        seq_fmt='seq',
        seq_suffix=None,    # default to seq_fmt
        ct_dir=None,        # default to seq_dir, then data_dir
        ct_fmt='ct',
        ct_suffix=None,
        lib_min_len=None,
        lib_max_len=None,
        seq2upper=True,     # all resnames to upper case
        check_unknown=True,
        check_duplicate=True,
        debug=False,
        tqdm_disable=False,
        **kwargs):
    """ build dataset from seq and ct file databases with cross checking """
    recap = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    # get seq and ct files
    seq_dir = misc.get_1st_value([seq_dir, ct_dir, data_dir], default='./')
    seq_dir = Path(seq_dir)
    seq_suffix = misc.get_1st_value([seq_suffix, seq_fmt], '.seq')
    if len(seq_suffix) and not seq_suffix.startswith('.'):
        seq_suffix = '.' + seq_suffix
    logger.info(f'Searching dir: {seq_dir} for **/*{seq_suffix}...')
    seq_files = list(seq_dir.glob('**/*' + seq_suffix))
    logger.info(f'Found {len(seq_files)} {seq_fmt} files')

    ct_dir = misc.get_1st_value([ct_dir, seq_dir, data_dir], default='./')
    ct_dir = Path(ct_dir)
    ct_suffix = misc.get_1st_value([ct_suffix, ct_fmt], '.ct')
    if len(ct_suffix) and not ct_suffix.startswith('.'):
        ct_suffix = '.' + ct_suffix
    logger.info(f'Searching dir: {ct_dir} for **/*{ct_suffix}...')
    ct_files = list(ct_dir.glob('**/*' + ct_suffix))
    logger.info(f'Found {len(ct_files)} {ct_fmt} files')

    gwio.files_check_sibling(seq_files, fdir=None if seq_dir.samefile(ct_dir) else ct_dir,
                suffix=ct_suffix)
    gwio.files_check_sibling(ct_files, fdir=None if ct_dir.samefile(seq_dir) else seq_dir,
                suffix=seq_suffix)

    if debug: seq_files = seq_files[:100]

    # main loop
    logger.info(f'Reading {len(seq_files)} {seq_fmt} files...')
    seqs_data = molstru.SeqStruct2D(sorted(seq_files), fmt=seq_fmt, database=data_base)

    recap.update(seq_dir=str(seq_dir), seq_fmt=seq_fmt,
                 ct_dir=str(ct_dir), ct_fmt=ct_fmt,
                 num_seq_files=len(seq_files), num_ct_files=len(ct_files),
                 num_seqs=len(seqs_data.seq))
    recap.len_minmax = [seqs_data.len.min(), seqs_data.len.max()]

    logger.info(f'Successfully parsed {recap.num_seqs} {seq_fmt} files, len_minmax: {recap.len_minmax}')

    seqs_data.idx = np.arange(1, recap.num_seqs + 1)
    if lib_min_len is not None and lib_min_len > recap.seq_minmax[0]:
        seqs_data.len_inrange = seqs_data.len > lib_min_len
    else:
        seqs_data.len_inrange = None

    if lib_max_len is not None and lib_max_len < recap.seq_minmax[1]:
        if seqs_data.len_inrange is None:
            seqs_data.len_inrange = seqs_data.len <= lib_max_len
        else:
            seqs_data.len_inrange = np.logical_and(seqs_data.selected, seqs_data.len <= lib_max_len)

    if seqs_data.len_inrange is None:
        recap.num_len_inrange = recap.num_seqs
        seqs_data.len_inrange = np.full(recap.num_seqs, True)
    else:
        recap.num_len_inrange = seqs_data.len_inrange.astype(int).sum()
        logger.info(f'{recap.num_len_inrange} out of {recap.num_seqs} are ' + \
                f'within the length range of [{lib_min_len}, {lib_max_len}]')

    cts_data = misc.Struct(num=np.zeros(recap.num_seqs, dtype=int), # the number of ct files found
                          len=np.zeros(recap.num_seqs, dtype=int), # should be the length of ct, not seq
                          seq=[''] * recap.num_seqs,
                          id=[''] * recap.num_seqs,
                          ct=[np.empty(0, dtype=int)] * recap.num_seqs,
                          sameSeq=np.full(recap.num_seqs, False),
                          sameSeqCase=np.full(recap.num_seqs, False),
                          )

    logger.info(f'Reading {ct_fmt} files...')
    for iseq, seq_file in enumerate(tqdm(seqs_data.file)):

        if not seqs_data.len_inrange[iseq]:
            logger.debug(f'Skipping out-of-length-range {seq_fmt}: {seq_file}')
            continue

        # get ct file
        if seq_dir.samefile(ct_dir):
            _ct_file = Path(seq_file).with_suffix(ct_suffix)
        else:
            _ct_file = ct_dir / (Path(seq_file).stem + ct_suffix)

        if _ct_file.exists():
            cts_data.num[iseq] = 1
        else:
            _ct_file = list(_ct_file.parent.glob(_ct_file.stem + '_p*' + ct_suffix))
            cts_data.num[iseq] = len(_ct_file)
            if len(_ct_file):
                logger.info(f'{len(_ct_file)} {ct_fmt} files found for {seq_file}')
                print(_ct_file)
                _ct_file = _ct_file[0]

        if cts_data.num[iseq] <= 0:
            logger.warning(f'No {ct_suffix} file found for {seq_file} in {ct_dir}')
            continue

        # load ct and check consistency
        if ct_fmt == 'ct':
            _ct_data = molstru.parse_ct_lines(_ct_file)
        elif ct_fmt == 'bpseq':
            _ct_data = molstru.parse_bpseq_lines(_ct_file)
        elif ct_fmt in ['st', 'sta']:
            # [0] is used here because a st file can contain multiple sequences
            _ct_data = molstru.parse_st_lines(_ct_file, fmt=ct_fmt)[0]
        elif ct_fmt in ['bps']:
            _ct_data = dict(
                id=str(_ct_file),
                ct=molstru.parse_bps_lines(_ct_file),
                seq=seqs_data.seq[iseq],
                resnum=np.linspace(1, seqs_data.len[iseq], seqs_data.len[iseq], dtype=int),
                )
            # _ct_data =
        else:
            logger.critical(f'Unrecognized ct_fmt: {ct_fmt}')

        # _ct_data['seq'] is saved only when there is conflict with seq_data.seq[iseq]
        cts_data.id[iseq] = _ct_data['id']
        cts_data.ct[iseq] = _ct_data['ct']
        cts_data.len[iseq] = len(_ct_data['ct']) # max([_ct_data['len'], len(_ct_data['seq'])])

        # only check residues in _ct_data['resnum']
        if _ct_data['resnum'][-1] <= seqs_data.len[iseq]:
            _seq_seq = ''.join([seqs_data.seq[iseq][_i - 1] for _i in _ct_data['resnum']])
        else:
            logger.warning(f'Larger resnum in {ct_fmt} than {seq_fmt}_len: {seq_file}')
            _seq_seq = ''.join([seqs_data.seq[iseq][_i - 1] for _i in
                            _ct_data['resnum'][_ct_data['resnum'] <= seqs_data.len[iseq]]])

        _ct_seq = ''.join([_ct_data['seq'][_i - 1] for _i in _ct_data['resnum']])

        if len(_seq_seq) != len(_ct_seq):
            logger.warning(f'Length mismatch of the sequence: {seq_file}')
        elif _seq_seq == _ct_seq:
            cts_data.sameSeq[iseq] = True
            cts_data.sameSeqCase[iseq] = True
        else:
            _seq_seq = _seq_seq.upper()
            _ct_seq = _ct_seq.upper()
            if _seq_seq == _ct_seq:
                cts_data.sameSeq[iseq] = True
                logger.info(f'Found seq case conflict: {seq_file}')
            elif seq_fuzzy_match(_ct_seq, _seq_seq):
                cts_data.sameSeq[iseq] = True
                logger.info(f'Found seq case conflict and fuzzy match: {seq_file}')
            else:
                logger.warning(f'[{seq_file}]: {seq_fmt} and {ct_fmt} have different sequence!')

        # # same seq and ct length
        # if seq_data.len[iseq] == len(_ct_data['seq']):
        #     if seq_data.seq[iseq] == _ct_data['seq']: # a perfect match!
        #         ct_data.sameSeq[iseq] = True
        #         ct_data.sameSeqCase[iseq] = True
        #     else: # check case and N
        #         _seq_seq = seq_data.seq[iseq].upper()
        #         _ct_seq = _ct_data['seq'].upper()

        #         if _seq_seq == _ct_seq:
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq case conflict for {seq_file}')
        #         elif seq_match_lookup(_ct_seq, _seq_seq):
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq lookup needed for {seq_file}')
        #         else:
        #             logger.warning(f'[{seq_file}]: ct has the same len but different seq!')

        # # different seq and ct length
        # elif seq_data.len[iseq] > _ct_data['len']:
        #     # check whether a subset of seqdata.seq[iseq]
        #     _seq_seq = ''.join([seq_data.seq[iseq][_i - 1] for _i in _ct_data['resnum']])
        #     _ct_seq = ''.join([_ct_data['seq'][_i - 1] for _i in _ct_data['resnum']])
        #     if _seq_seq == _ct_seq:
        #         ct_data.sameSeq[iseq] = True
        #         ct_data.sameSeqCase[iseq] = True
        #     else:
        #         _seq_seq = _seq_seq.upper()
        #         _ct_seq = _ct_seq.upper()
        #         if _seq_seq == _ct_seq:
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq case conflict for {seq_file}')
        #         elif seq_match_lookup(_ct_seq, _seq_seq):
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq lookup needed for {seq_file}')
        #         else:
        #             logger.warning(f'[{seq_file}]: ct has different length and sequence!')

        if not cts_data.sameSeq[iseq]:
            cts_data.seq[iseq] = _ct_data['seq']
            print(f'[{_ct_file}]: SEQ: {seqs_data.len[iseq]} and CT: {_ct_data["len"]}')
            print(f'Sequence mismatch for {_ct_file} -->')
            print(f'SEQ: {seqs_data.seq[iseq]}')
            print(f' CT: {cts_data.seq[iseq]}')

    logger.info(f'Finished reading {cts_data.num.sum()} {ct_fmt} files')

    # collect results and analyze

    logger.info(f'Collecting data as dataframes for post-processing...')
    seqs_data.ct = cts_data.ct
    df_all = seqs_data.to_df(seq=True, ct=True, has=True,
                            res_count=True, bp_count=True,
                            stem_count=True, pknot_count=True)
    df_all = df_all.assign(
            # data info
            lenInrange = seqs_data.len_inrange,
            numCT = cts_data.num,
            lenCT = cts_data.len,
            seqCT = cts_data.seq,
            sameSeqCT = cts_data.sameSeq,
            sameSeqCaseCT = cts_data.sameSeqCase,
            # fields for resolving duplicates
            unknownSeq = False,
            conflictSeq = np.logical_and(np.logical_not(cts_data.sameSeq), seqs_data.len_inrange),
            duplicateSeq = np.full(recap.num_seqs, False),
            conflictVal = False,
            idxSameSeq = [0] * recap.num_seqs,
            idxSameSeqVal = [0] * recap.num_seqs,
            # fields for saving pkl and lib
            save2lib = np.logical_and(cts_data.sameSeq, seqs_data.len_inrange),
            )

    recap.num_len_inrange = df_all.lenInrange.sum()
    recap.num_seq_matched = df_all.sameSeqCT.sum()
    recap.num_case_conflict = recap.num_seq_matched - df_all.sameSeqCaseCT.sum()

    count_all_residues(df_all, args=recap, prefix='raw_')

    if seq2upper:
        logger.info('Converting to upper cases...')
        df_all.seq = misc.mpi_map(str.upper, tqdm(df_all.seq, desc='Convert2Upper'), quiet=True )

    if check_unknown:
        check_unknown_residues(df_all, recap=recap)

    if check_duplicate:
        check_duplicate_keyvals(df_all, recap=recap, keys='seq', vals=['ct'])

    # kwargs['save_prefix'] = kwargs.get('save_prefix', auto_save_prefix)
    save_files(df_all, args=recap, tqdm_disable=tqdm_disable, **kwargs)


def gather_rna2d(
        r2d_files=None,              # can be one or multiple files readable by molstru.SeqStruct2D()
        data_base=None,              # database for extracting moltype: archiveII, stralign, bpRNA, etc.
        data_dir=None, r2d_dir=None, # r2d_dir overrides data_dir (default: ./)
        data_fmt=None, r2d_fmt=None, # r2d_fmt overrides data_fmt (default: ct))
        data_size=None,              # random sample size (None: all)
        r2d_suffix=None,             # file suffix (default: .r2d_fmt)
        idx_start=1,                 # start idx for all files (default: 1)
        filename2idx=False,          # extract idx as file.stem.split('_')[0]
        idx2filename=False,          # whether add idx_ to file and id
        idx_digits=11,               # number of digits for idx (default: 11)
        min_len=None,                # min length of seq for down-stream processing and libset
        max_len=None,                # max length of seq for down-stream processing and libset
        seq2upper=False,             # resnames to upper case
        check_unknown=False,         # check for unknown residues
        check_duplicate=False,       # check for duplicate sequences
        save_lumpsum=True,           # whether to save aggregated file(s)
        save_lib=True,               # whether to save the libset (within [min_len, max_len])
        save_pkl=True,               # whether to save aggregated pkl file
        save_csv=True,               # whether to save aggregated csv file
        csv_exclude=['ct', 'ctmat', 'bpmat', 'ppm', 'dist'], # exclude these cols for csv files
        tqdm_disable=False,          # NOTE: see save_files for additional saving args!!!
        **kwargs):
    """ build dataset from RNA sequence and 2D structure files
    Note:
        file names are rectified by gwio.str2filename()
        idx_ is added to each file.stem and id
        save2lib considers 1) len_inrange and 2) not unknownSeq
    Todo:
        all duplicating sequences are kept for the raw and lib sets.
        maybe check for consistency at some point and remove duplicates.
        well, conflicting structures are probably more realistic, or assign partial base pairs
    """
    args = misc.Struct(locals())
    logger.debug('Arguments:\n' + gwio.json_str(args.__dict__))

    # get r2d_files
    if r2d_files is not None and len(r2d_files):
        if isinstance(r2d_files, str):
            r2d_files = [r2d_files]
        r2d_files = [Path(_f) for _f in r2d_files]
        r2d_fmt = misc.get_1st_value([r2d_fmt, data_fmt, r2d_files[0].suffix[1:]], default='ct')

        auto_save_prefix = f'{r2d_files[0].stem}_brewed' if len(r2d_files) == 1 else \
            f'{r2d_files[0].stem}-{r2d_files[-1].stem}-N{len(r2d_files)}_brewed'
    else:
        r2d_dir = misc.get_1st_value([r2d_dir, data_dir], default='./')
        r2d_fmt = misc.get_1st_value([r2d_fmt, data_fmt], default='ct')
        r2d_suffix = misc.get_1st_value([r2d_suffix, r2d_fmt], default='.ct')
        if len(r2d_suffix) and not r2d_suffix.startswith('.'):
            r2d_suffix = '.' + r2d_suffix

        auto_save_prefix = Path(r2d_dir).resolve().stem
        
        # r2d_files = [_f.resolve().as_posix() for _f in data_dir.glob('**/*.ct')]
        logger.info(f'Searching dir: {r2d_dir} for **/*{r2d_suffix}...')
        r2d_files = list(Path(r2d_dir).glob('**/*' + r2d_suffix))
        logger.info(f'Found {len(r2d_files)} {r2d_fmt} files')

    if data_size is not None:
        logger.info(f'Selecting data_size: {data_size} ...')
        r2d_files = r2d_files[:data_size]

    # read ct files and apply length range [min_len, max_len]
    logger.info(f'Reading {len(r2d_files)} {r2d_fmt} files...')
    r2d_gizmo = molstru.SeqStruct2D(sorted(r2d_files), fmt=r2d_fmt, database=data_base)
    args.num_files = len(r2d_files)
    args.num_seqs = len(r2d_gizmo.seq)
    args.len_minmax = [r2d_gizmo.len.min(), r2d_gizmo.len.max()]
    logger.info(f'Successfully parsed {args.num_seqs} {r2d_fmt} files, len_minmax: {args.len_minmax}')

    if min_len is not None and min_len > args.len_minmax[0]:
        r2d_gizmo.len_inrange = r2d_gizmo.len > min_len
    else:
        r2d_gizmo.len_inrange = None

    if max_len is not None and max_len < args.len_minmax[1]:
        if r2d_gizmo.len_inrange is None:
            r2d_gizmo.len_inrange = r2d_gizmo.len <= max_len
        else:
            r2d_gizmo.len_inrange = np.logical_and(r2d_gizmo.len_inrange, r2d_gizmo.len <= max_len)

    if r2d_gizmo.len_inrange is None:
        args.num_len_inrange = args.num_seqs
        r2d_gizmo.len_inrange = np.full(args.num_seqs, True)
    else:
        args.num_len_inrange = r2d_gizmo.len_inrange.astype(int).sum()
        logger.info(f'{args.num_len_inrange} out of {args.num_seqs} are ' + \
                f'within length range of [{min_len}, {max_len}]')

   # collect results and analyze
    logger.info(f'Collecting data into dataframes to scrutinize...')

    seq_len = np.array([len(_s) for _s in r2d_gizmo.seq], dtype=int)
    df_all = r2d_gizmo.to_df(idx_start=idx_start,
                            seq=True, ct=True, dbn=True,
                            ssn=r2d_fmt in ['st'],
                            nkn=r2d_fmt in ['st'],
                            tangle=r2d_fmt in ['tangle'],
                            has=True,
                            res_count=True, bp_count=True,
                            stem_count=True, pknot_count=True)

    # convert all file name by gwio.str2filename()
    df_all['srcfile'] = df_all['file']
    df_all['file'] = [gwio.str2filename(Path(_s).name) for _s in df_all['file']]
    df_all['srcid'] = df_all['id']
    df_all['id'] = [_s.strip() for _s in df_all['id']]

    if idx2filename and filename2idx:
        logger.error(f'idx2filename and filename2idx cannot be both True')
    elif idx2filename:
        # get the strings from idx
        if idx_digits is not None and isinstance(idx_digits, int):
            idx_strs = [f'{_i:0{idx_digits}d}' for _i in df_all['idx']]
        else:
            idx_strs = [f'{_i}' for _i in df_all['idx']]
        df_all['file'] = [f'{idx_strs[_i]}_{df_all.iloc[_i]["file"]}' for _i in range(len(df_all))]
        df_all['id'] = [f'{idx_strs[_i]}_{df_all.iloc[_i]["id"]}' for _i in range(len(df_all))]
    elif filename2idx:
        try:
            df_all['idx'] = [int(Path(_s).stem.split('_')[0]) for _s in df_all['file']]
        except:
            logger.error(f'Failed to extract idx from filenames')
    else:
        logger.debug(f'idx2filename and filename2idx are both False')

    # df_all['lenBin'] = df_all['len'] // bin_len if bin_len else 0

    df_all = df_all.assign(
            lenDiffSeq = r2d_gizmo.len - seq_len,
            # fields for resolving duplicates
            unknownSeq = False,
            conflictSeq = r2d_gizmo.len != seq_len,
            duplicateSeq = False, # np.full(args.num_seqs, False),
            idxSameSeq = -1, # [-1] * args.num_seqs,
            idxSameSeqVal = -1, # [-1] * args.num_seqs,
            conflictVal = False, # np.full(args.num_seqs, False),
            # fields for saving pkl and lib
            lenInrange = r2d_gizmo.len_inrange,
            save2lib = r2d_gizmo.len_inrange,
            # dataset = [''] * args.num_seqs, # 'train' or 'valid'
            )
    args.num_len_mismatch = (df_all.lenDiffSeq != 0).sum()

    args.src_shape = df_all.shape
    count_all_residues(df_all, args=args, prefix='src_', show=False)
    args.src_columns = df_all.columns.tolist()

    if seq2upper:
        logger.info('Converting to upper cases...')
        df_all.seq = misc.mpi_map(str.upper, tqdm(df_all.seq, desc='Seq2Upper'), quiet=True)

    if check_unknown:
        check_unknown_residues(df_all, recap=args)

    df_all['save2lib'] = np.logical_and(df_all['save2lib'], ~ df_all['unknownSeq'])

    if check_duplicate:
        check_duplicate_keyvals(df_all, recap=args, keys='seq', vals=['ct'])

    # get logical_and between save2lib and conflictSeq
    # df_all['save2lib'] = np.logical_and(df_all['save2lib'], ~ df_all['conflictSeq'])

    args.out_shape = df_all.shape
    count_all_residues(df_all, args=args, prefix='out_', show=False)
    args.out_columns = df_all.columns.tolist()
    
    kwargs.setdefault('save_prefix', auto_save_prefix)
    kwargs.setdefault('save_fasta', True)
    save_files(df_all, args=args, save_lumpsum=save_lumpsum, save_pkl=save_pkl, save_csv=save_csv,
               csv_exclude=csv_exclude, save_lib=save_lib, tqdm_disable=tqdm_disable, **kwargs)


def gather_fasta(fasta_files=None, **kwargs):
    """ a wrapper for gather_seq with seq_fmt as fasta """
    args = dict(seq_fmt='fasta', )
    args.update(kwargs)
    gather_seq(seq_files=fasta_files, **args)


def gather_seq(
        seq_files=None,
        data_base=None,                 # achiveII/stralign/bprna, used to get mol_type
        dir=None, data_dir=None, seq_dir=None,
        fmt=None, data_fmt=None, seq_fmt=None,
        suffix=None, seq_suffix=None,   # default: .seq_fmt
        data_size=None,                 # limit the data_size (usually for testing)
        has_upp=False,                  # whether has upp (unused)
        min_len=None,                   # remove seqs with len < min_len
        max_len=None,                   # remove seqs with len > max_max
        idx_start=1,                    # start idx (default: 1)
        idx_digits=None,                # number of digits for idx (default: None)
        idx2filename=None,              # add idx_ to file and id
        filename2idx=None,              # extract idx from filename as filename.stem.split('_')[0]
        id2idx=False,                   # extract idx from id as id.split('_')[0]
        seq2upper=False,                # convert to upper case
        seq2dna=False,                  # convert to DNA alphabets (ATGC) (not yet implemented)
        check_unknown=False,            # check for unknown resnames
        check_duplicate=False,          # check for duplicated seqs
        tqdm_disable=False,             # NOTE: see save_all_files for saving args!!!
        **kwargs):
    """ build dataset from seq file database only """
    args = misc.Struct(locals())
    logger.debug('Arguments:\n' + gwio.json_str(args.__dict__))

    seq_fmt = misc.get_1st_value([seq_fmt, data_fmt, fmt], default='fasta')
    if seq_files is not None and len(seq_files):
        if isinstance(seq_files, str): seq_files = [seq_files]
        seq_files = [Path(_f) for _f in seq_files]
        auto_save_prefix = seq_files[0].stem
    else:
        seq_dir = misc.get_1st_value([seq_dir, data_dir, dir], default='./')
        seq_suffix = misc.get_1st_value([seq_suffix, suffix, seq_fmt])
        if len(seq_suffix) and not seq_suffix.startswith('.'):
            seq_suffix = '.' + seq_suffix
        seq_dir = Path(seq_dir)

        logger.info(f'Searching dir: {seq_dir} for **/*{seq_suffix}...')
        seq_files = list(seq_dir.glob('**/*' + seq_suffix))
        logger.info(f'Found {len(seq_files)} {seq_fmt} files')

        auto_save_prefix = Path(seq_dir).resolve().stem

    if data_size is not None:
        logger.info(f'Selecting data_size: {data_size} ...')
        seq_files = seq_files[:data_size]

    # read files and apply length range [min_len, max_len]
    logger.info(f'Reading {len(seq_files)} {seq_fmt} files...')
    seqsData = molstru.SeqStruct2D(sorted(seq_files), fmt=seq_fmt, database=data_base)

    args.update(data_dir=data_dir, seq_dir=seq_dir, seq_fmt=seq_fmt)
    args.num_files = len(seq_files)
    args.num_seqs = len(seqsData.seq)
    args.len_minmax = [seqsData.len.min(), seqsData.len.max()]
    logger.info(f'Successfully parsed {args.num_seqs} {seq_fmt} sequences')

    seqsData.idx = np.arange(1, args.num_seqs + 1)
    if min_len is not None and min_len > args.seq_minmax[0]:
        seqsData.len_inrange = seqsData.len > min_len
    else:
        seqsData.len_inrange = None

    if max_len is not None and max_len < args.seq_minmax[1]:
        if seqsData.len_inrange is None:
            seqsData.len_inrange = seqsData.len <= max_len
        else:
            seqsData.len_inrange = np.logical_and(seqsData.selected, seqsData.len <= max_len)

    if seqsData.len_inrange is None:
        args.num_len_inrange = args.num_seqs
        seqsData.len_inrange = np.full(args.num_seqs, True)
    else:
        args.num_len_inrange = seqsData.len_inrange.astype(int).sum()
        logger.info(f'{args.num_len_inrange} out of {args.num_seqs} are ' + \
                f'within length range of [{min_len}, {max_len}]')

   # collect results and analyze
    logger.info(f'Collecting data as dataframe to scrutinize...')

    seq_len = np.array([len(_s) for _s in seqsData.seq], dtype=int)

    df_all = seqsData.to_df(idx_start=idx_start,
                            seq=True, has=True, res_count=True)

    if id2idx and filename2idx:
        logger.error(f'Cannot have both id2idx and filename2idx')
    elif id2idx:
        df_all['idx'] = [int(_s.split('_')[0]) for _s in df_all['id']]

    if idx2filename and filename2idx:
        logger.error(f'Cannot have both idx2filename and filename2idx')
    elif idx2filename:
        df_all['srcfile'] = df_all['file']
        df_all['srcid'] = df_all['id']
        # get the strings from idx
        if idx_digits is not None and isinstance(idx_digits, int):
            idx_strs = [f'{_i:0{idx_digits}d}' for _i in df_all['idx']]
        else:
            idx_strs = [f'{_i}' for _i in df_all['idx']]
        df_all['file'] = [f'{idx_strs[_i]}_{misc.str_deblank(Path(df_all.iloc[_i]["file"]).name)}' for _i in range(len(df_all))]
        df_all['id'] = [f'{idx_strs[_i]}_{df_all.iloc[_i]["id"].strip()}' for _i in range(len(df_all))]        
    elif filename2idx:
        try:
            df_all['idx'] = [int(Path(_s).stem.split('_')[0]) for _s in df_all['file']]
        except:
            logger.error(f'Failed to extract idx from filenames')        


    df_all = df_all.assign(
            lenInrange = seqsData.len_inrange,
            # lenDiffSeq = seqsData.len - seq_len,
            # fields for resolving duplicates
            # unknownSeq = False,
            # conflictSeq = seq_data.len != seq_len,
            # duplicateSeq = np.full(args.num_seqs, False, dtype=bool),
            # idxSameSeq = [0] * args.num_seqs,
            # idxSameSeqVal = [0] * args.num_seqs,
            # conflictVal = False,
            # fields for saving pkl and lib
            save2lib = seqsData.len_inrange,
            # dataset = [''] * args.num_seqs, # 'train' or 'valid'
            )

    count_all_residues(df_all, args=args, prefix='src_')

    if seq2upper:
        logger.info('Converting to upper cases...')
        df_all.seq = misc.mpi_map(str.upper, tqdm(df_all.seq, desc='Seq2Upper'), quiet=True )

    if seq2dna:
        df_all.seq = misc.mpi_map(molstru.seq2DNA, tqdm(df_all.seq, desc='Seq2DNA'), quiet=True )
        # logger.critical(f'seq2dna is not implemented yet!!!')

    if check_unknown:
        check_unknown_residues(df_all, recap=args)

    if check_duplicate:
        check_duplicate_keyvals(df_all, recap=args, keys='seq', vals=None)

    args.out_shape = df_all.shape
    count_all_residues(df_all, args=args, prefix='all_')
    args.out_columns = df_all.columns.tolist()

    kwargs.setdefault('save_lumpsum', True)
    kwargs.setdefault('save_pkl', True)
    kwargs.setdefault('save_prefix', auto_save_prefix) # inspect.currentframe().f_code.co_name)
    save_files(df_all, args=args, tqdm_disable=tqdm_disable, **kwargs)


def gather_upp2021(data_names=None, data_dir='./', beam_size=1000, bp_cutoff=0.0,
            save_dir=None, save_pickle=True, save_pkl=True, save_lib=True,
            tqdm_disable=False, **kwargs):
    """ build the 2021 upp data from paddle"""

    from slurp_python import launch_linear_rna as linear_rna

    data_dir = Path(data_dir)
    save_dir = Path(save_dir) if save_dir else data_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if data_names is None:
        data_names = ['train', 'valid', 'test', 'predict']
    elif isinstance(data_names, str):
        data_names = [data_names]

    for fname in data_names:
        logger.info(f'Reading data type: {fname}...')
        seqsdata = molstru.SeqStruct2D(data_dir / (fname + '.txt'),
                fmt='fasta', dbn=True, upp=(fname not in ['predict', 'test']))
        logger.info(f'Successfully parsed {len(seqsdata.seq)} sequences')

        # compute dbn and bpp with linear_fold and linear_partition
        partial_func = functools.partial(linear_rna.linear_fold_v, beam_size=beam_size)

        dbn_v = misc.mpi_map(partial_func, seqsdata.seq)

        # [0]: dbn, [1]: energy
        dbn_v = [_v[0] for _v in dbn_v]

        partial_func = functools.partial(linear_rna.linear_partition_c, beam_size=beam_size, bp_cutoff=bp_cutoff)
        # [0]: partition function, [1]: base pairing probabilities
        bpp_c = misc.mpi_map(partial_func, seqsdata.seq)
        upp_c = []
        for i, bpp in enumerate(bpp_c):
            if len(bpp[1]):
                upp_c.append(molstru.bpp2upp(np.stack(bpp[1], axis=0), seqsdata.len[i]))
            else:
                upp_c.append(np.zeros(seqsdata.len[i], dtype=np.float32))
                logger.info(f'No bpp_c found for seq: {seqsdata.seq[i]}')
        # upp_c = itertools.starmap(mol_stru.bpp2upp, zip(bpp_c, seqsdata.len))

        partial_func = functools.partial(linear_rna.linear_partition_v, beam_size=beam_size, bp_cutoff=bp_cutoff)
        bpp_v = misc.mpi_map(partial_func, seqsdata.seq)

        upp_v = []
        for i, bpp in enumerate(bpp_v):
            if len(bpp[1]):
                upp_v.append(molstru.bpp2upp(np.stack(bpp[1], axis=0), seqsdata.len[i]))
            else:
                upp_v.append(np.zeros(seqsdata.len[i], dtype=np.float32))
                logger.info(f'No bpp_v found for seq: {seqsdata.seq[i]}')
        # upp_c = itertools.starmap(mol_stru.bpp2upp, zip(bpp_c, seqsdata.len))

        upp_pred = [np.stack((upp_c[_i], upp_v[_i]), axis=1) \
                        for _i in range(len(upp_c))]

        df = seqsdata.to_df(seq=True, ct=True, upp=True, has=True, res_count=True)
        csv_file = save_dir / (fname + '.csv')
        logger.info(f'Storing csv: {csv_file}')
        df.to_csv(csv_file, index=False, float_format='%8.6f')

        if save_pickle and save_pkl:
            midata = dict(id=seqsdata.id, idx=seqsdata.idx, len=seqsdata.len, seq=seqsdata.seq,
                        dbn=dbn_v if len(dbn_v) else seqsdata.dbn,
                        file=seqsdata.file)

            if len(upp_pred):
                logger.info('Add predicted upp values as extra in midata')
                midata['extra'] = upp_pred

            if fname not in ['test', 'predict']:
                midata['upp'] = seqsdata.upp

            pkl_file = save_dir / (fname + '.pkl')
            logger.info(f'Storing pickle: {pkl_file}')
            gwio.pickle_squeeze(midata, pkl_file, fmt='lzma')

        if save_lib:
            lib_dir = save_dir / fname
            lib_dir.mkdir(exist_ok=True)
            logger.info(f'Saving lib files in: {lib_dir}')

            with tqdm(total=len(seqsdata.seq), disable=tqdm_disable) as prog_bar:
                for i in range(len(seqsdata.seq)):
                    lib_file = lib_dir / f'{i + 1:d}.input.fasta'
                    with lib_file.open('w') as iofile:
                        iofile.writelines('\n'.join(
                            seqsdata.get_fasta_lines(i, dbn=True, upp=False)
                        ))

                    if fname not in ['test', 'predict']:
                        lib_file = lib_dir / f'{i + 1:d}.label.upp'
                        np.savetxt(lib_file, seqsdata.upp[i], '%8.6f')
                        # with lib_file.open('w') as iofile:
                        #     iofile.writelines('\n'.join(seqsdata.upp[i].astype(str)))

                    if (i + 1) % 10 == 0:
                        prog_bar.update(10)
                prog_bar.update((i + 1) % 10)


if __name__ == '__main__':
    misc.argv_fn_caller(sys.argv[1:]) # module=sys.modules[__name__], verbose=1)
