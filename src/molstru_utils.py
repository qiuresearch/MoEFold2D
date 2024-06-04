# coding: utf-8
# Note that Biopython needs to be 1.72 for save atoms to work! 
# For biopython to work for nucleic acids, modify the following:
# 1) in Bio/SeqUtils/__init__.py, seq1(), simply return the input seq
#    the purpose of seq1() is to convert 3-letter to 1-letter, but it is not needed for nucleic acids
#    we now reserve the original names for nucleic acids such as A, DA, C, DC, G, DG, U, DT, etc.
# 2) in Bio/SeqIO/PdbIO.py, for both PdbSeqresIterator and CifSeqresIterator (three places total)
#    change record = SeqRecord(Seq(",".join(residues))) to
#    record = SeqRecord(Seq(",".join(residues))

import gzip
import json
import logging
import os
import socket
import urllib
from collections import Counter
from io import StringIO
from pathlib import Path
from re import S
from shutil import copy2

import xpdb
from Bio import SeqIO
# from yaml import parse
from Bio.PDB import PDBIO, MMCIFParser, PDBList, PDBParser, Select

# disable proxy
urllib.request.install_opener(
    urllib.request.build_opener(
        urllib.request.ProxyHandler({})
        )
    )
import fetch_pdb
# home brew
import misc
import molstru
import molstru3d
import molstru_config

logger = logging.getLogger(__name__)

def read_fasta_dbn(fname,  fsuffix='', fdir='') :
    """ read fasta or dbn file and return a dict """
    fname = str(fname)
    ss_info = dict()
    dbnfile = os.path.join(fdir, fname+fsuffix)
    if not os.path.isfile(dbnfile):
        dbnfile = os.path.splitext(dbnfile)[0] + '.dbn'
        if not os.path.isfile(dbnfile):
            dbnfile = os.path.splitext(dbnfile)[0] + '.fasta'
            if not os.path.isfile(dbnfile) :
                logger.warning('Cannot find fasta or dbn file for {}'.format(dbnfile))
                return ss_info

    with open(dbnfile) as hfile :
        txtlines = hfile.readlines()
    # strip and remove empty lines
    txtlines = [oneline.strip() for oneline in txtlines]
    txtlines = list(filter(None, txtlines))

    if len(txtlines) < 2 :
        print('Less than two lines in the fasta/dbn file')
        return ss_info

    ss_info['title'] = txtlines[0][1:]
    # check for breaks in the fasta and dbn
    ss_info['seqRaw'] = txtlines[1]
    ss_info['seq'] = ss_info['seqRaw'].replace('&', '')
    ind_breaks = misc.str_find_all(ss_info['seqRaw'], '&')
    ss_info['iseqNewFrag'] = [0] + [ibreak - i for i, ibreak in enumerate(ind_breaks)]
    seqNewFrags = ss_info['iseqNewFrag'] + [len(ss_info['seq'])]
    ss_info['lenFrags'] =  [ seqNewFrags[ifrag+1]-seqnum for ifrag, seqnum in enumerate(ss_info['iseqNewFrag'])]
    ss_info['numBreaks'] = len(ind_breaks)

    if len(txtlines) >= 3 : # has dbn
        ss_info['dbnRaw'] = txtlines[2]
        ss_info['dbn'] = ''.join([dbn for dbn in txtlines[2] if dbn != '&'])
        if len(ss_info['dbn']) != len(ss_info['seq']) :
            logger.warning('Different SEQ and DBN lengths for {}'.format(fname))

    return ss_info

def seqres2list(seqres) :
    seqres_list = []
    # post process the seqre_pdb, whose .seq is comma separated original letters
    # WARNING: you need to use my modified version of biopython to get such .seq
    # Each element is a dict() for one chain, with structure {id=pdbid:chainid,
    # name=, seq=, dbxrefs=[], annotations={'chain':, 'molecule_type':}}
    for _seqres in seqres:
        chain_dict = dict()
        chain_dict['id'] = _seqres.id
        chain_dict['chain'] = _seqres.annotations['chain']
        chain_dict['moltype'] = _seqres.annotations['molecule_type'] # not sure why it is always protein
        chain_dict['dbxrefs'] = _seqres.dbxrefs
        chain_dict['seqRaw'] = [str(_s) for _s in _seqres.seq.split(',')]
        chain_dict['seq'] = molstru_config.res2seqcode(chain_dict['seqRaw'], restype='auto')
        chain_dict['features'] = _seqres.features
        chain_dict['_per_letter_annotations'] = _seqres._per_letter_annotations
        seqres_list.append(chain_dict)
    return seqres_list

def read_rcsb_by_rcsbid(rcsbid, rcsb_dir='') :
    """ read and/or download a pdb/cif file and return Biopython structure and seqres dict"""

    pdb_stru, pdb_seqres = None, None

    # try PDB first
    rcsbid = rcsbid.upper()
    pdbfile = os.path.join(rcsb_dir, f'{rcsbid}.pdb')
    # check divided data structure
    if not os.path.isfile(pdbfile):
        rcsbid = rcsbid.lower()
        pdbfile = os.path.join(rcsb_dir, f'{rcsbid}.pdb')
        if not os.path.isfile(pdbfile):
            pdbfile = os.path.join(rcsb_dir, 'pdb', rcsbid[1:3], f'pdb{rcsbid}.ent.gz')

    if os.path.isfile(pdbfile):
        logger.info("Reading PDB file: {}".format(pdbfile))
        pdb_stru, pdb_seqres = read_pdb_file(rcsbid, pdbfile, structure_builder=None)
        if pdb_stru is None:
            logger.info(f'Parsing PDB file again with xpdb.SloppyStructureBuilder()...')
            pdb_stru, pdb_seqres = read_pdb_file(rcsbid, pdbfile, structure_builder=xpdb.SloppyStructureBuilder())

    if pdb_stru is not None:
        return pdb_stru, pdb_seqres

    # try mmCIF
    rcsbid = rcsbid.upper()
    ciffile = os.path.join(rcsb_dir, f'{rcsbid}.cif')
    if not os.path.isfile(ciffile) :
        rcsbid = rcsbid.lower()
        ciffile = os.path.join(rcsb_dir, f'{rcsbid}.cif')
        if not os.path.isfile(ciffile) :
            ciffile = os.path.join(rcsb_dir, 'mmCIF', rcsbid[1:3], f'{rcsbid}.cif.gz')

    if not os.path.isfile(ciffile):
        # this will check obsolete records
        logger.debug('Try again with fetch_pdb.fetch_cif to download the file in PDBx/mmCIF format ...')
        ciffile = fetch_pdb.fetch_cif(rcsbid, outdir=rcsb_dir, unzip=False)

    if not os.path.isfile(ciffile):
        logger.debug('Attempt to download the file in PDBx/mmCIF format ...')
        try:
            pdb_wget = PDBList()
            ciffile = pdb_wget.retrieve_pdb_file(rcsbid, pdir=rcsb_dir)
        except Exception as err:
            logger.critical(f'Failed to download: {ciffile} !!!')
            print(err)

    if os.path.isfile(ciffile) :
        pdb_stru, pdb_seqres = read_cif_file(rcsbid, ciffile, structure_builder=None)
        if pdb_stru is None:
            logger.info(f'Parsing cif file again with xpdb.SloppyStructureBuilder()...')
            pdb_stru, pdb_seqres = read_cif_file(rcsbid, ciffile, structure_builder=xpdb.SloppyStructureBuilder())

    if pdb_stru is None:
        logger.critical("Unable to download/parse PDB or CIF for ID: {}".format(rcsbid))
        return None, None
    return pdb_stru, pdb_seqres

def parse_modres_lines(pdblines, modres_dict={}, check_na=True):
    """ returns a list of modres_dicts, one item per chain
    check_na=True check against known modmap """
    # modres_lines =
    for aline in pdblines:
        if aline.startswith(('ATOM', 'HETATM')):
            break
        if not aline.startswith('MODRES'):
            continue
        # not sure if strip() breaks the convention
        resname = aline[12:15].strip()
        chainid = aline[16]
        resid = aline[18:23] # inclue Insertion code aline[22]
        stdname = aline[24:27].strip()
        resdesc = aline[29:]

        # check against known modmaps
        if stdname != modres_dict.get(resname, stdname):
            logger.critical(f'Inconsistent stdname for {resname}, PDB:{stdname}, modres_dict:{modres_dict[resname]}!!!')

        if check_na and stdname != molstru_config.NA_ResNameModMap.get(resname, stdname):
            logger.critical(f'Inconsistent stdname for {resname}, PDB:{stdname}, molstru_config:{molstru_config.NA_ResNameModMap[resname]}!!!')

        modres_dict[resname] = stdname
    return modres_dict

def read_pdb_file(pdbid, pdbfile, structure_builder=None):
    """ return pdb_stru and pdb_seqres """
    if pdbfile.endswith('gz'):
        with gzip.open(pdbfile, 'rt') as iofile:
            pdblines = iofile.read()
    else:
        with open(pdbfile, 'r') as iofile:
            pdblines = iofile.read()

    pdbparser = PDBParser(QUIET=True, PERMISSIVE=True, structure_builder=structure_builder)
    try:
        pdb_stru = pdbparser.get_structure(pdbid.upper(), StringIO(pdblines))
        seqres_list = seqres2list(SeqIO.parse(StringIO(pdblines), 'pdb-seqres'))
        # seqres_list['modres'] = parser_modres_lines(pdblines.split('\n'))
        return pdb_stru, seqres_list
        # with open(pdbfile, 'r') as iofile:
        #     pdb_stru = pdbparser.get_structure(pdbid.upper(), iofile)
        #     iofile.seek(0)
        #     pdb_seqres = seqres2dict(SeqIO.parse(iofile, 'pdb-seqres'))
    except Exception as err:
        logger.critical(f'An error occurred when parsing pdb: {pdbfile} !!!')
        print(err)
        return None, None

def read_cif_file(pdbid, ciffile, structure_builder=None):
    """  """
    cifparser = MMCIFParser(QUIET=True, structure_builder=structure_builder)
    # xpdb.SloppyStructureBuilder())
    logger.info("Reading CIF file: {}".format(ciffile))

    try:
        if ciffile.endswith('.gz'):
            with gzip.open(ciffile, 'rt') as iofile:
                pdb_stru = cifparser.get_structure(pdbid, iofile)
                iofile.seek(0)
                pdb_seqres = seqres2list(SeqIO.parse(iofile, 'cif-seqres'))
        else:
            with open(ciffile, 'r') as iofile:
                pdb_stru = cifparser.get_structure(pdbid, iofile)
                iofile.seek(0)
                pdb_seqres = seqres2list(SeqIO.parse(iofile, 'cif-seqres'))
        return pdb_stru, pdb_seqres
    except Exception as err:
        logger.critical(f'An error occurred when parsing cif: {ciffile} !!!')
        print(err)
        return None, None

class RNASelect(Select):
    """ Define a class to select RNA residues only """
    def accept_residue(self, residue):
        if molstru_config.ResNames_RNA.find(residue.resname.strip().upper()) != -1:
            return True
        else:
            return False
    def accept_atom(self, atom):
        if atom.is_disordered():
            return False
        else:
            return True

def get_pdb_by_rcsbinfo(rcsbinfo, rcsb_dir='', save_dir='', save_rcsb_recap=True, NA_moltype=True, **DUMMY) :
    ''' pdbinfo is a string of "RCSBid|model#|chainID" (: or _ can replace |)
        multiple chains can be specified with a separator '+'
        empty model# or chainid outputs all models or chains

        Each saved PDB file contains one or more chains in only ONE model, though
        it is designed to extract one chain in one model from the RCSB PDB

        Todo:
            1) accept model number as * and chainid as *
            2) accept passing modelnum and chainid as arguments
            3) current design is really only for extract single or multiple chains
                from the same PDBID and model

        Warning: the 2-letter chain ids in CIF are converted into single letters for PDB
    '''
    rcsb_recap = {'id':rcsbinfo, 'fileStem':''}
    rcsbinfo_list = rcsbinfo.replace(':','_').replace('|','_').split('+')

    rcsbid_old, rcsb_stru, seqres_list, save_stru = None, None, None, None
    chainid2sav_all, seqres_chainids, rcsbid = [], [], '' # pdbid should be just the 4-letter code
    for rcsbinfo in rcsbinfo_list : # each entry is processed and added to atoms2sav
        modelnum, chainid = '', ''
        # parse pdbinfo to get pdbid, modelnum, chainid
        if rcsbinfo.count('_') >= 2 :
            [rcsbid, modelnum, chainid] = rcsbinfo.split('_')
        elif rcsbinfo.count('_') == 1 :
            [rcsbid, modelnum] = rcsbinfo.split('_')
            if not str.isnumeric(modelnum):
                chainid = modelnum
                modelnum = ''
        else:
            rcsbid = rcsbinfo
        if modelnum != '' : modelnum = int(modelnum)

        rcsbid = rcsbid.strip()
        logger.debug('Processing RCSBId: {}, Model: {}, Chain: {}'.format(rcsbid, modelnum, chainid))
        # read the pdb if needed
        if rcsbid != rcsbid_old :
            rcsb_stru, seqres_list = read_rcsb_by_rcsbid(rcsbid, rcsb_dir=rcsb_dir)
            if rcsb_stru is None:
                return None, rcsb_recap, None

            # get some header information (such as resolution, etc)
            rcsb_recap['rcsbType'] = rcsb_stru.header['head']
            rcsb_recap['rcsbTitle'] = rcsb_stru.header['name']
            rcsb_recap['rcsbResolution'] = rcsb_stru.header['resolution'] if rcsb_stru.header['resolution'] else -1
            rcsb_recap['rcsbMethod'] = rcsb_stru.header['structure_method'].upper()
            rcsb_recap['rcsbDepositDate'] = rcsb_stru.header.get('deposition_date', '')
            rcsb_recap['rcsbReleaseDate'] = rcsb_stru.header.get('release_date', '')
            rcsb_recap['rcsbAuthor'] = rcsb_stru.header.get('author')
            rcsb_recap['rcsbKeywords'] = rcsb_stru.header.get('keywords')

            # collect some stats
            rcsb_recap['rcsbNumModels'] = len(rcsb_stru.child_list)
            rcsb_recap['rcsbNumChains'] = len(rcsb_stru[0].child_list)
            pdb_resnames = [_resi.resname for _resi in rcsb_stru[0].get_residues()]
            pdb_restypes, pdb_moltype, rescounts = count_residue_types(pdb_resnames, prefix='rcsb', NA_moltype=False)
            rcsb_recap['rcsbMoltype'] = pdb_moltype
            rcsb_recap.update(rescounts)
            # for stru_chain in stru_pdb[imod].child_list :
                # pdb_report['srcNumResidues'] += len(stru_chain.child_list)

            if 'missing_residues' in rcsb_stru.header:
                rcsb_recap['rcsbNumMissingRes'] = len(rcsb_stru.header['missing_residues'])
                # the list is converted to str so as to be assigned to a pandas dataframe
                rcsb_recap['rcsbMissingRes'] = str([ [ '' if _res['model'] is None else _res['model'],
                                                _res['chain'],
                                                _res['res_name'],
                                                _res['ssseq'],
                                                ' ' if _res['insertion'] is None else _res['insertion'] ]
                                            for _res in rcsb_stru.header['missing_residues']])
            else :
                rcsb_recap['rcsbNumMissingRes'] = 0
                rcsb_recap['rcsbMissingRes'] = '[]'
                # pdb_report['srcMissingModel'] = str([_res['model'] for _res in stru_pdb.header['missing_residues']])
                # pdb_report['srcMissingModel'] = pdb_report['srcMissingModel'].replace('None', "''")
                # pdb_report['srcMissingChain'] = str([_res['chain'] for _res in stru_pdb.header['missing_residues']])
                # pdb_report['srcMissingResname'] = str([_res['res_name'] for _res in stru_pdb.header['missing_residues']])
                # pdb_report['srcMissingResnum'] = str([_res['ssseq'] for _res in stru_pdb.header['missing_residues']])
                # pdb_report['srcMissingResicode'] = str([_res['insertion'] for _res in stru_pdb.header['missing_residues']])
                # pdb_report['srcMissingResicode'] = pdb_report['srcMissingResicode'].replace('None', "' '")

            # process the seqres info
            rcsb_recap['rcsbHasSeqRes'] = False
            rcsb_recap['rcsbUndefSeqRes'] = []
            for _seqres in seqres_list :
                rcsb_recap['rcsbUndefSeqRes'] += [_seqres['seqRaw'][_i]
                    for _i, _seq in enumerate(_seqres['seq']) if _seq == '?' ]

            rcsb_recap['rcsbUndefSeqRes'] = '|'.join(rcsb_recap['rcsbUndefSeqRes'])
            seqres_chainids = [_seqres['chain'] for _seqres in seqres_list]

            # save the FULL pdb seqres in JSON and FASTA formats
            if save_rcsb_recap:
                seqres_file = os.path.join(rcsb_dir, rcsbid+'.seqres')
                seqres_lines = []
                for _seqres in seqres_list :
                    seqres_lines.append('>'+rcsbid + "_"+_seqres['chain'])
                    seqres_lines.append(''.join(_seqres['seq']))
                    if len(_seqres['seq']) > 0:
                        rcsb_recap['rcsbHasSeqRes'] = True
                if len(seqres_lines):
                    with open(seqres_file, 'w') as iofile :
                        iofile.write('\n'.join(seqres_lines))

                json_file = os.path.join(rcsb_dir, rcsbid+'.json')
                with open(json_file, 'w') as iofile:
                    json.dump(rcsb_recap, iofile, indent=3)

        rcsbid_old = rcsbid

        # create an empty structure for saving
        if save_stru is None :
            save_stru = rcsb_stru.copy()
            for imod in rcsb_stru.child_dict.keys() :
                save_stru.detach_child(imod)

        # get modelnum(s) to save
        if modelnum == '' :
            logger.debug('No model number specified, all models saved!!!')
            modelnum = list(rcsb_stru.child_dict.keys())
        else :
            modelnum = [max(0,modelnum-1)] # have seen modelnum = 0 from pdbInfo!

        # process each model to be saved
        for imod in modelnum :
            if imod not in rcsb_stru.child_dict :
                logger.warning('Modelnum: {} not found, skipping...'.format(imod))
                continue

            # create an empty model
            if imod not in save_stru.child_dict :
                save_stru.insert(imod, rcsb_stru[imod].copy())
                for _chainid in rcsb_stru[imod].child_dict.keys() :
                    save_stru[imod].detach_child(_chainid)

            # get chainid(s) to save
            if chainid == '' :
                logger.debug('No chain ID specified, all chains will be saved!!!')
                chainid2sav_model = list(rcsb_stru[imod].child_dict.keys())
            elif chainid in rcsb_stru[imod].child_dict :
                chainid2sav_model = [chainid]
            else :
                logger.debug('ChainID: {} not found!!! ALL CHAINS SAVED!!!'.format(chainid))
                chainid2sav_model = list(rcsb_stru[imod].child_dict.keys())
            chainid2sav_all += chainid2sav_model

            # insert chains to the saved model and get new chainid if needed
            chainid_used = list(save_stru[imod].child_dict.keys())
            for _chainid in chainid2sav_model :
                stru_chain = rcsb_stru[imod][_chainid].copy()
                chainid_new = molstru_config.get_unique_chainid(stru_chain.id, used_ids=chainid_used)
                if len(chainid_new) == 0 :
                    logger.warning('Too many chains to save, skipping chain: {}'.format(stru_chain.id))
                    # chainid_new = [chainid_used[-1]]
                    continue
                chainid_used.append(chainid_new[0])
                if stru_chain.id not in seqres_chainids :
                    logger.warning('Chain {} not in SEQRES record for RCSBid: {}'.format(stru_chain.id, rcsbid))
                if rcsb_recap['rcsbNumMissingRes'] > 0 and stru_chain.id != chainid_new[0]:
                    rcsb_recap['rcsbMissingRes'] = rcsb_recap['rcsbMissingRes'].replace(f"'{stru_chain.id}'", f"'{chainid_new[0]}'")
                stru_chain.id = chainid_new[0]
                save_stru[imod].insert(len(chainid_used)-1, stru_chain)

    # check...
    if save_stru is None :
        logger.warning('No PDB structure to be saved!!!')
        return None, rcsb_recap, None

    # extract some information (using the first model only!!!)
    save_model = save_stru.child_list[0]
    rcsb_recap['saveNumChains'] = len(save_model.child_list)
    save_resnames = [_resi.resname for _resi in save_model.get_residues()]
    save_restypes, save_moltype, rescounts = count_residue_types(save_resnames, prefix='save', NA_moltype=NA_moltype)
    rcsb_recap['saveMoltype'] = save_moltype
    rcsb_recap.update(rescounts)
    rcsb_recap['saveNumMissingRes'] = 0
    rcsb_recap['saveMissingRes'] = []
    if rcsb_recap['rcsbNumMissingRes'] > 0 :
        for _res in eval(rcsb_recap['rcsbMissingRes']): # _res: [modelnum, chainid, resname, resnum, altloc?]
            if _res[1] in save_model.child_dict.keys() :
                rcsb_recap['saveNumMissingRes'] += 1
                rcsb_recap['saveMissingRes'].append(_res)
    rcsb_recap['saveMissingRes'] = str(rcsb_recap['saveMissingRes'])

    # saving files
    save_pdbfiles = ['']
    if True:
        pdbio = PDBIO()
        if save_dir and not os.path.exists(save_dir) : os.makedirs(save_dir)
        for save_model in save_stru.get_models():
            rcsb_recap['fileStem'] = molstru_config.pdbinfo2filename(rcsb_recap['id'],
                modelnum=save_model.id+1, chainid=None) # list(stru_mod.child_dict.keys()))
            save_pdbfiles.append(os.path.join(save_dir, rcsb_recap['fileStem']+'.pdb'))

            pdbio.set_structure(save_model)
            with open(save_pdbfiles[-1], 'w') as iofile :
                pdbio.save(iofile)
            logger.info('Saving pdb file: {}'.format(save_pdbfiles[-1]))

            seqres_file = os.path.join(save_dir, rcsb_recap['fileStem'] + '.seqres')
            seqres_sav, seqres_lines, seqres_seq, seqres_num= [], [], [], 0
            for _chainid in chainid2sav_all: # stru_mod.child_dict.keys(): # chainid is forced to be one letter in stru_mod
                if _chainid not in seqres_chainids : continue
                iseqres = seqres_chainids.index(_chainid)
                seqres_sav.append(seqres_list[iseqres])
                seqres_lines.append('>' + molstru_config.pdbinfo2filename(rcsbid, modelnum=save_model.id+1, chainid=_chainid))
                seqres_lines.append(''.join(seqres_list[iseqres]['seq']))
                seqres_seq.append(seqres_lines[-1])
                seqres_num += len(seqres_list[iseqres]['seq'])
            if len(seqres_lines) > 1 :
                rcsb_recap['saveHasSeqRes'] = True
                with open(seqres_file, 'w') as iofile : iofile.write('\n'.join(seqres_lines))
            else :
                rcsb_recap['saveHasSeqRes'] = False
            rcsb_recap['saveNumSeqRes'] = seqres_num
            rcsb_recap['saveSeqRes'] = '|'.join(seqres_seq)

            report_file = os.path.join(save_dir, rcsb_recap['fileStem']+'.json')
            with open(report_file, 'w') as iofile :
                pdb_report_sav = rcsb_recap.copy()
                pdb_report_sav.update({'saveSeqResList':str(seqres_sav)})
                # print(pdb_report_sav)
                json.dump(pdb_report_sav, iofile, indent=3)

    return save_stru, rcsb_recap, save_pdbfiles[-1]

def get_moltype(rescounts, NA_only=True, numRes=None, numRNA=None, numDNA=None, numAA=None):
    """ Determine moltype from residue counts

    Keyword args can overwrite recounts vals

    Two modes of operation:
        1) NA_only=True (default)
            returns one of RNA, dRNA, NA, rDNA, DNA, and UNK (if no NA residues)
        2) NA_only=False
            returns one of AA, AA??NA??, NA, and UNK. (?? is the number percentage)

    TODO:
        1) deal with AA
        2) deal with ModiNA
    """
    # if numRes is None: numRes = rescounts['Res']
    # if numAA is None: numAA = rescounts['NA']

    if numDNA is None: numDNA = rescounts['DNA']
    if numRNA is None: numRNA = rescounts['RNA']
    numNA = numRNA + numDNA + rescounts['ModiNA']

    if NA_only:
        if numNA > 0:
            rna_ratio = (numRNA + rescounts['ModiNA']) / numNA
            moltype = \
                'RNA' if rna_ratio == 1.0 else \
                'dRNA' if rna_ratio >= 0.8 else \
                'NA' if rna_ratio > 0.2 else \
                'rDNA' if rna_ratio > 0 else \
                'DNA'
        else:
            moltype = 'UNK'
            logger.warning(f'Cannot determine moltype !!!')
    else:
        if numAA is None: numAA = rescounts['AA']
        numRes = numAA + numNA + rescounts['ModiNA']
        if numRes > 0:
            aa_ratio = numAA / numRes * 100
            if aa_ratio == 100:
                moltype = 'AA'
            elif aa_ratio > 0:
                # aa_ratio = int(aa_ratio // 10 * 10)
                aa_ratio = int(aa_ratio)
                moltype = f'AA{aa_ratio:02d}NA{100-aa_ratio:02d}'
            else:
                moltype = 'NA'
        else:
            moltype = 'UNK'
            logger.warning(f'Cannot determine moltype !!!')

    return moltype

def count_residue_types(resnames, prefix=None, NA_moltype=True,
        AA_ResName3List=molstru_config.AA_ResName3List,
        DNA_ResName2List= molstru_config.DNA_ResName2List,
        RNA_ResNameList= molstru_config.RNA_ResNameList,
        RNA_ResName3List= molstru_config.RNA_ResName3List,
        NA_ResNameModMap=molstru_config.NA_ResNameModMap,
        Ligand_ResNamesList=molstru_config.Ligand_ResNamesList,
        ):
    """ return the types (RNA, DNA, AA, etc.) of each residue and
    a dict of counts. resnames can be an iterator
    NA_moltype=True passes NA_only=True to get_moltype()
    """
    restypes = []
    for resname in resnames:
        resname = resname.upper().strip()
        if resname == '':
            restypes.append('NULL')
        elif resname in RNA_ResNameList or resname in RNA_ResName3List:
            restypes.append('RNA')
        elif resname in DNA_ResName2List:
            restypes.append('DNA')
        elif NA_ResNameModMap.get(resname, None): # modified (Checked no overlap between Standard AAs and NA_ResNameModMap)
            restypes.append('ModiNA')
        elif resname in AA_ResName3List: # modified AAs have many overlap with NA_ResNameModMap
            restypes.append('AA')
        elif resname in Ligand_ResNamesList:
            restypes.append('Ligand')
        else:
            restypes.append('Unknown')
            logger.debug(f'Unknown resname: {resname} !!!')

    rescounts = Counter(restypes)
    rescounts['Res'] = len(restypes)
    rescounts['NA'] = rescounts['RNA'] + rescounts['DNA'] + rescounts['ModiNA']
    moltype = get_moltype(rescounts, NA_only=NA_moltype)

    report = {}
    prefix = f'{prefix}Num' if prefix else 'num'
    keys = ['Res', 'NA', 'RNA', 'DNA', 'ModiNA', 'AA', 'Ligand', 'Empty', 'Unknown']
    for key in keys:
        report[f"{prefix}{key}"] = rescounts[key]

    return restypes, moltype, report

def get_atom_lines(pdbfile, pdbdir=None) :
    """ a snippet for getting atom_lines """
    if isinstance(pdbfile, str):
        if pdbdir is not None:
            pdbfile = os.path.join(pdbdir, pdbfile)
        if not os.path.exists(pdbfile):
            pdbfile = pdbfile + '.pdb'
        if not os.path.exists(pdbfile) :
            logger.warning(f"Cannot find PDB file: {pdbfile} !!!")
            atom_lines = []
        with open(pdbfile, 'r') as ofile:
            atom_lines = ofile.readlines()
        # No stripping, keeping \n at the end
        atom_lines = [strline for strline in atom_lines \
                    if ('ATOM  ' == strline[0:6]) or ('HETATM' == strline[0:6])]
    elif isinstance(pdbfile, list):
        atom_lines = pdbfile.copy()
    else :
        logger.warning("Neither a filename or list of strings was passed, returning!!!")
        atom_lines = []
    return atom_lines

def get_seq_from_atom_lines(pdbfile, pdbdir=None):
    ''' simply get the sequence from a PDB file or a list of ATOM/HETATM lines.
        resnames are stripped!!!
    '''
    atom_lines = get_atom_lines(pdbfile, pdbdir=pdbdir)

    if len(atom_lines):
        resnames = [atom_lines[0][17:20].strip()]
        resnums = [misc.str2int(atom_lines[0][22:26])] #col27 is insertion code
        iatom_newresi = [0]
    else:
        return [], [], [], []

    for i in range(1, len(atom_lines)) :
        # include the insertion code in comparison
        if atom_lines[i][22:27] != atom_lines[i-1][22:27]:
            resnames.append(atom_lines[i][17:20].strip())
            resnums.append(misc.str2int(atom_lines[i][22:26]))
            iatom_newresi.append(i)

    return resnames, resnums, iatom_newresi, atom_lines

def select_pdb_residues(pdbfile, startnum=1, endnum=None, resnum_list=None, savefile=None):
    ''' Select residue numbers starting from "startnum or by seqno".
        This is a very simple function only processing ATOM and HETATM lines!
    '''
    _, pdb_seq_num, iloc_newseq, atom_lines = get_seq_from_atom_lines(pdbfile)

    # generate new sequence numbers unless passed
    if resnum_list == None :
        if endnum == None :
            endnum = len(pdb_seq_num)
        resnum_list = pdb_seq_num[startnum-1:endnum]

    # selecting sequence numbers in seqnum
    new_atom_lines = []
    iloc_newseq.append(len(atom_lines))
    for iseq in range(0,len(iloc_newseq)-1) :
        if pdb_seq_num[iseq] in resnum_list :

            new_atom_lines.extend(atom_lines[iloc_newseq[iseq]:iloc_newseq[iseq+1]])

    if isinstance(savefile, str) :
        # shutil.copyfile(pdbfile, pdbfile+'.old')
        with open(pdbfile, 'w') as ofile:
            ofile.write(''.join(new_atom_lines))

    return new_atom_lines

def renumber_atom_lines(pdbfile, resnum_start=1, atomnum_start=-1,
        atomnum_list=None, resnum_list=None, savefile=None) :
    ''' Renumber residue numbers starting from "resnum_start or using resnum_list".
        This is a very simple function only processing ATOM and HETATM lines!
    '''
    _, _, iloc_newseq, atom_lines = get_seq_from_atom_lines(pdbfile)

    # generate new sequence numbers
    if resnum_list == None :
        resnum_list = list(range(resnum_start, len(iloc_newseq)+resnum_start))

    # assign new sequence numbers
    iloc_newseq.append(len(atom_lines))
    for iseq in range(0,len(iloc_newseq)-1):
        for i in range(iloc_newseq[iseq], iloc_newseq[iseq+1]):
            # this also removes the altLoc and chain insertion code
            atom_lines[i] = 'ATOM  ' + atom_lines[i][6:16]+' '+atom_lines[i][17:22]+\
                            ('%4i ' % resnum_list[iseq]) + atom_lines[i][27:]

    if isinstance(savefile, str) :
        # shutil.copyfile(pdbfile, pdbfile+'.old')
        with open(pdbfile, 'w') as ofile:
            ofile.write(''.join(atom_lines))

    return atom_lines

def remove_alternate_locations(pdbfile,  savefile=None):
    '''     '''
    _, _, iloc_newseq, atom_lines = get_seq_from_atom_lines(pdbfile)

    # selecting sequence numbers in seqnum
    new_atom_lines = []
    num_removed_atoms = 0
    iloc_newseq.append(len(atom_lines))
    for iseq in range(0,len(iloc_newseq)-1) :
        # find the unique atom names in the residue
        atom_names = []
        for iatom in range(iloc_newseq[iseq], iloc_newseq[iseq+1]) :
            if atom_lines[iatom][12:16] not in atom_names :
                atom_names.append(atom_lines[iatom][12:16])
                new_atom_lines.append(atom_lines[iatom])
            else :
                num_removed_atoms += 1

    if (num_removed_atoms > 0) and isinstance(savefile, str) :
        # shutil.copyfile(pdbfile, pdbfile+'.old')
        with open(pdbfile, 'w') as ofile:
            ofile.write(''.join(new_atom_lines))

    return new_atom_lines

def bake_rna_lib(src_pdb, pdbid='', save_dir='', 
        seq=None, seqres=None, # missing_res=None,
        keep_protein=False, keep_dna=True, keep_rna=True, keep_modina=True,
        keep_ligand=False, keep_unknown=False, keep_empty=False,
        keep_altloc=False, rename_modina=True,
        rename_hetatm=True, renumber_residue=False, renumber_atom=False,
        write_pdb=True, pdb_suffix='.pdb', write_fasta=True, fasta_suffix='.seqatm', **DUMMY):
    '''Start from a source PDB, clean up, and collect statistics
    
    Consider using numpy 2D string array in the future, a bit tedious now

    Attempt to fix a single PDB chain based on seq, seqres, and missing residues
    seq - from nt.seqres extracted from pdb_seqres.txt
    seqres - from the SEQRES recored parsed by Biopython
    missing_res - from missing_residues parsed by Biopython

    Task:
        1) Identify the

    '''
    pdbid = Path(src_pdb).stem

    rna_report = {
        # 'sameSeqSeqRes':None, 'sameSeqSeqResLen':None, 'sameSeqSeqResFuzzy':None,
        # 'srcMoltype': '', 'srcNumResTotal':0, 'srcNumResModi':0, 'srcNumResLigands':0,
        # 'srcNumResUnknown':0, 'srcNumResEmpty':0, 'srcNumAA':0, 'srcNumDNA':0, 'srcNumRNA':0,
        # 'srcNumResInserts':0, 'srcNumResNumBreaks':0, 'srcNumAtomAltlocs':0, 'srcNumHetatms':0,
        # 'srcResModi':set(), 'srcResLigands':set(), 'srcResUnknown':set(),
        # 'libNumResTotal':0, 'libNumAA':0, 'libNumDNA':0, 'libNumRNA':0,
        # 'libNumResModi':0, 'libMoltype':'', 'isLibPDBSaved':False,
        }

    if seq is not None and seqres is not None:
        if len(seq) == len(seqres):
            rna_report['srcSameSeqSeqResLen'] = True
            rna_report['srcSameSeqSeqRes'] = seq == seqres
            if rna_report['srcSameSeqSeqRes']:
                rna_report['srcSameSeqSeqResFuzzy'] = True
            else:
                rna_report['srcSameSeqSeqResFuzzy'] = molstru.seq_fuzzy_match(seq, seqres)
        else:
            rna_report['srcSameSeqSeqResLen'] = False
            rna_report['srcSameSeqSeqRes'] = False
            rna_report['srcSameSeqSeqResFuzzy'] = False
    else:
            rna_report['srcSameSeqSeqRes'] = None
            rna_report['srcSameSeqSeqResLen'] = None
            rna_report['srcSameSeqSeqResFuzzy'] = None

    resnames, resnums, iatom_newres, atom_lines = get_seq_from_atom_lines(src_pdb)

    # check src_pdb restypes
    restypes, moltype, rescounts = count_residue_types(
        resnames, prefix='src',
        RNA_ResNameList = molstru_config.RNA_ResNameList, # + 'N'
        RNA_ResName3List = molstru_config.RNA_ResName3List,
        DNA_ResName2List = molstru_config.DNA_ResName2List, # + ['DN']
        NA_ResNameModMap = molstru_config.NA_ResNameModMap
        )
    rna_report['srcMoltype'] = moltype
    rna_report.update(rescounts)

    # collect some stats about src_pdb and rename ModiNA=True residues
    rna_report['srcModiNA'] = set()
    for _i, _restype in enumerate(restypes):
        if _restype != 'ModiNA': continue
        rna_report['srcModiNA'].add(resnames[_i])
        if rename_modina:
            resnames[_i] = molstru_config.NA_ResNameModMap[resnames[_i]]

    rna_report['srcModiNA'] = '|'.join(rna_report['srcModiNA'])
    rna_report['srcLigand'] = '|'.join(set([
        resnames[_i] for _i, _restype in enumerate(restypes) if _restype == 'Ligand'
        ]))
    rna_report['srcUnknown'] = '|'.join(set([
        resnames[_i] for _i, _restype in enumerate(restypes) if _restype == 'Unknown'
        ]))

    # get the residues for lib_pdb
    kept_restypes = []
    if keep_protein: kept_restypes.append('AA')
    if keep_dna: kept_restypes.append('DNA')
    if keep_rna: kept_restypes.append('RNA')
    if keep_modina: kept_restypes.append('ModiNA')
    if keep_ligand: kept_restypes.append('Ligand')
    if keep_unknown: kept_restypes.append('Unknown')
    if keep_empty: kept_restypes.append('NULL')

    if len(kept_restypes) == 0:
        logger.warning('No residue types selected to keep!!!')

    ires2rna = [_i for _i, _restype in enumerate(restypes) if _restype in kept_restypes]
    rna_restypes = [restypes[_i] for _i in ires2rna]
    rna_resnames = [resnames[_i] for _i in ires2rna]
    rna_resnums = [resnums[_i] for _i in ires2rna]

    rna_rescounts = Counter(rna_restypes)
    rna_rescounts['Res'] = len(rna_restypes) # this is the total
    rna_rescounts['NA'] = rna_rescounts['RNA'] + rna_rescounts['DNA'] + rna_rescounts['ModiNA']
    rna_report['libMoltype'] = get_moltype(rna_rescounts, NA_only=True)
    keys = ['Res', 'NA', 'RNA', 'DNA', 'ModiNA', 'AA', 'Ligand', 'Empty', 'Unknown']
    for key in keys:
        rna_report[f"libNum{key}"] = rna_rescounts[key]

    seqatm = ''.join(rna_resnames)
    rna_report['libSeqAtm'] = seqatm

    if seq is not None:
        if len(seq) == len(seqatm):
            rna_report['libSameSeqSeqAtmLen'] = True
            rna_report['libSameSeqSeqAtm'] = seq == seqatm
            if rna_report['libSameSeqSeqAtm']:
                rna_report['libSameSeqSeqAtmFuzzy'] = True
            else:
                rna_report['libSameSeqSeqAtmFuzzy'] = molstru.seq_fuzzy_match(seq, seqatm)
        else:
            rna_report['libSameSeqSeqAtmLen'] = False
            rna_report['libSameSeqSeqAtm'] = False
            rna_report['libSameSeqSeqAtmFuzzy'] = False
    else:
            rna_report['libSameSeqSeqAtm'] = None
            rna_report['libSameSeqSeqAtmLen'] = None
            rna_report['libSameSeqSeqAtmFuzzy'] = None

    # check for chain breaks and residue inserts
    dif_resnum_list = [rna_resnums[i]-rna_resnums[i-1] for i in range(1,len(rna_resnums))]

    rna_report['libNumResInserts'] = dif_resnum_list.count(0)
    rna_report['libNumResNumBreaks'] = len([i for i in dif_resnum_list if abs(i)>1])

    # save atoms and check for altloc
    rna_report['libNumAtomAltlocs'] = 0
    rna_report['libNumHetatms'] = 0
    iatom_newres.append(len(atom_lines)) # be careful that size changes!!!
    rna_iatom_newres, rna_atom_lines = [], []

    running_atom_num, running_resi_num = 0, 0
    for new_iseq, iseq in enumerate(ires2rna):
        rna_iatom_newres.append(len(rna_atom_lines))
        running_resi_num += 1

        atom_altloc = '' # keeps the name of the atom with altloc set
        # new resname removing altLoc (#17) and chain insertion code (#27)
        for i in range(iatom_newres[iseq], iatom_newres[iseq+1]):
            # check for altloc atoms and remove >=2nd atoms if not keep_altloc
            if atom_lines[i][16] != ' ' :
                rna_report['libNumAtomAltlocs'] += 1
                if (not keep_altloc) :
                    if atom_altloc == atom_lines[i][12:16]:
                        continue
                    else :
                        atom_altloc = atom_lines[i][12:16] # remember the first atom

            # if len(atom_lines[i]) < 80 :
                # print('Length:{}[{}]'.format(len(atom_lines[i]), atom_lines[i]))
                # continue
            if atom_lines[i].startswith('HETATM'):
                rna_report['libNumHetatms'] += 1
            running_atom_num += 1
            # removes the insertion code
            rna_atom_lines.append( ('ATOM  ' if rename_hetatm else atom_lines[i][0:6]) +
                ('{:5d} '.format(running_atom_num) if renumber_atom else atom_lines[i][6:12]) +
                atom_lines[i][12:16] + (atom_lines[i][16] if keep_altloc else ' ') +
                '{:>3s}'.format(rna_resnames[new_iseq]) + atom_lines[i][20:22] +
                ('{:4d} '.format(running_resi_num) if renumber_residue else atom_lines[i][22:27]) +
                atom_lines[i][27:] )
            # only residue name is changed in the line below
            # new_atom_lines.append(atom_lines[i][0:17]+'{:>3s}'.format(resname)+atom_lines[i][20:])

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if write_pdb:
        with open(os.path.join(save_dir, pdbid+pdb_suffix), 'w') as hfile :
            hfile.writelines(rna_atom_lines)
        rna_report['libPDBSaved'] = True

    if write_fasta:
        with open(os.path.join(save_dir, pdbid+fasta_suffix), 'w') as hfile :
            hfile.writelines(['>{}\n'.format(pdbid), ''.join(rna_resnames)])

    return rna_atom_lines, rna_report, rna_resnames, rna_resnums, rna_iatom_newres

def dssr_read_json(jsonfile):
    """ read dssr json file and check integrity """
    try:
        with open(jsonfile, 'r') as hfile:
            jsondict = json.load(hfile)
    except:
        logger.error(f'Error reading JSON file: {jsonfile} !!!')
        return None

    if 'nts' not in jsondict.keys() :
        logger.critical(f'Json file: {jsonfile} is incomplete or corrupted!!!')
        return None
    return jsondict

def dssr_dbn_from_json(jsondict, seq=None):
    """ get the dbn info from json """
    
    if isinstance(jsondict, (str, Path)):
        jsondict = dssr_read_json(jsondict) 
    if jsondict is None:
        return None

    if seq is None:
        # dbn is likely corrupted if there are residues unrecognized by DSSR!!!
        if 'dbn' not in jsondict or 'all_chains' not in jsondict['dbn'] or 'sstr' not in jsondict['dbn']['all_chains']:
            logger.error(f'JSON info need to have ["dbn"]["all_chains"]["sstr"] fields!!!')
            return None
        return jsondict['dbn']['all_chains']['sstr']
    else:
        logger.error('not yet implemented')
        return jsondict['dbn']['all_chains']['sstr']

def fix_dssr_dbn_by_refseq_todo(fname, fdir='', save_dir='', write_dbn=False,
        dbn_suffix='.dbn', seq_suffix='.seqatm', json_suffix='.json', backup_suffix='_orig.dbn') :
    """ adding missing residues in the dbn file, with the rna seqatm record as the template.

    IMPORTANT: DSSR uses the sequence number in the PDB file, which SHOULD start from 1 !!!

    DSSR outputs the correspondence between the recognized nucleotide bases and the residue numbers,
    which serves as the key information for adding the missing residues in dssr outputs.
    This usually occurs at the ends, either an unrecognizable or dangling base

    ONE PROBLEM is that it doesnot know whether to insert the unrecognized residues to the
    segment before or after a break"""

    # read dbn files, better to save two lists, one for seq, one for dbn
    dbnfile = os.path.join(fdir, fname+dbn_suffix)
    if not os.path.exists(dbnfile) :
        logger.warning(f'Cannot find DBN file: {dbnfile}, returning!!!')
        return
    dbn_info = read_fasta_dbn(dbnfile)
    seq_dbn = list(dbn_info['seqRaw'])
    dbn_dbn = list(dbn_info['dbnRaw'])

    # read fasta and create a list
    seqatmfile = os.path.join(fdir, fname + seq_suffix)
    seqatm_info = read_fasta_dbn(seqatmfile)
    seq_seqatm = list(seqatm_info['seq'])

    # read dssr json
    jsonfile = os.path.join(fdir, fname+json_suffix)
    try :
        with open(jsonfile, 'r') as hfile :
            json_info = json.load(hfile)
    except :
        logger.warning(f'Error with dssr json: {jsonfile}, returning!!!')
        return seq_dbn, dbn_dbn

    if seqatm_info['seq'] == dbn_info['seq'] :
        print(f'Same {seq_suffix} and dbn sequences for {fname}, skipping...')
        return seq_dbn, dbn_dbn

    # resnum mapping from dssr json output
    iseq2resnum = [resi['nt_resnum'] for resi in json_info['nts']]
    if len(iseq2resnum) != (len(seq_dbn) - dbn_info['numBreaks']) :
        print('Mismatching json and dbn files for {}, returning!!!'.format(fname))
        return seq_dbn, dbn_dbn

    iseq_dbn = -1 # this points to the ith actual sequence in seq_raw/dnb_raw
    seq_new, dbn_new = [], []
    iseq2resnum.insert(0,0) # to catch the start
    for i, dbn_code in enumerate(dbn_dbn) :
        if dbn_code == '&' :  # just add to it
            seq_new.append(seq_dbn[i])
            dbn_new.append(dbn_dbn[i])
            continue

        # append all in-between
        iseq_dbn += 1
        resnum_start = iseq2resnum[iseq_dbn]
        resnum_end = iseq2resnum[iseq_dbn+1]

        seq_new += seq_seqatm[resnum_start:resnum_end]
        dbn_new += ['.']*(resnum_end-resnum_start-1) + list(dbn_dbn[i])

    # get the end if missing
    if iseq2resnum[-1] < len(seq_seqatm) :
        seq_new += seq_seqatm[iseq2resnum[-1]:]
        dbn_new += ['.']*(len(seq_seqatm) - iseq2resnum[-1])

    if write_dbn and (seq_new != dbn_info['seq']):
        logger.info(f'Saving fixed dbn file: {dbnfile} ... (old vs new shown below)')
        print(dbn_info['seqRaw'])
        print(''.join(seq_new))
        print(dbn_info['dbnRaw'])
        print(''.join(dbn_new))
        dbnfile_backup = dbnfile.replace('.dbn', backup_suffix)
        if not os.path.exists(dbnfile_backup) : copy2(dbnfile, dbnfile_backup)
        with open(dbnfile, 'w') as hfile :
            hfile.writelines('\n'.join(['>'+fname + '_fixed nts={}'.format(len(seq_new)-seq_new.count('&')),
                ''.join(seq_new), ''.join(dbn_new)]))

    return seq_new, dbn_new

def defrag_stru_by_res2res_dist(fname, fdir='', write_files=True, max_length=3, dist_cutoff=10,
                    pdb_suffix='.pdb', dbn_suffix='.dbn', seqatm_suffix='.seqatm',
                    backup_suffix='_orig') :
    """ removing short discontinuous fragments in the PDB and DBN files """
    pdb_file = os.path.join(fdir,fname+pdb_suffix)
    dbn_file = os.path.join(fdir, fname + dbn_suffix)
    seqatm_file = os.path.join(fdir, fname + seqatm_suffix)

    atoms_stru = molstru.AtomsData(pdb_file)
    atoms_seq = ''.join(atoms_stru.seqResidNames)

    dbn_dict = read_fasta_dbn(dbn_file)

    if 'seq' in dbn_dict:
        if atoms_seq != dbn_dict['seq'] :
            logger.warning("Inconsistent DBN and PDB sequences, please check!!!")
            print("ATM:", atoms_seq)
            print("DBN:", dbn_dict['seq'])
    else:
        dbn_dict['seq'] = atoms_seq

    if 'dbn' not in dbn_dict:
        dbn_dict['dbn'] = '.' * len(dbn_dict['seq'])

    dbn_dict['title'] = dbn_dict.get('title', fname)

    atoms_stru.calc_res2res_distance(neighbor_only=True)
    idx2rmv = atoms_stru.remove_dangling_residues(max_length=max_length, dist_cutoff=dist_cutoff)

    if idx2rmv:
        print('Removing dangling residues: {}'.format(idx2rmv))
        # correct the dbn file (Breaks removed in the field 'seq'!)
        dbn_dict['seq'] = ''.join(atoms_stru.seqResidNames)
        if 'dbn' in dbn_dict :
            try:
                dbn_dict['dbn'] = ''.join([dbn_dict['dbn'][_i] for _i in range(0, len(dbn_dict['dbn'])) if _i not in idx2rmv])
            except:
                logger.critical(f'idx2rmv:{idx2rmv}, dbn: {dbn_dict["dbn"]}')
        else :
            dbn_dict['dbn'] = '.' * len(dbn_dict['seq'])

        if write_files :
            pdb_backup = os.path.join(fdir, fname+backup_suffix+pdb_suffix)
            if not os.path.exists(pdb_backup) : copy2(pdb_file, pdb_backup)
            atoms_stru.write_pdb(pdb_file)
            atoms_stru.write_fasta(seqatm_file)

        if write_files and os.path.exists(dbn_file) :
            dbn_backup = os.path.join(fdir, fname+backup_suffix+dbn_suffix)
            if not os.path.exists(dbn_backup) : copy2(dbn_file, dbn_backup)
            with open(dbn_file, 'w') as ofile :
                ofile.write('>{}\n'.format(fname))
                ofile.write(dbn_dict['seq']+'\n')
                ofile.write(dbn_dict['dbn'])

    return atoms_stru, dbn_dict, idx2rmv

def fix_rna_lib(fname, lib_dir='', src_dir='', max_length=3, dist_cutoff=10,
        backup_suffix='_orig.pdb'):
    """
        This should be called after dssr has been run over the target PDB file!
        0) remove short dangling fragments (could be ligand-like nucleotide)
        1) finalize the fasta file as the "official" primary sequence
        2) rectify the dbn file to have the same sequence
        3) renumber the PDB file to have the same resnums as the fasta seq """

    lib_report = dict(libSeqSource=None)

    # 0) remove dangling fragments first
    pdb_file = os.path.join(lib_dir,fname+'.pdb')
    dbn_file = os.path.join(lib_dir, fname + '.dbn')
    seqatm_file = os.path.join(lib_dir, fname + '.seqatm')

    atoms, dbn_dict, idx2rmv = defrag_stru_by_res2res_dist(fname, fdir=lib_dir,
            write_files=True, max_length=max_length, dist_cutoff=dist_cutoff)
    atoms_seq = ''.join(atoms.seqResidNames)
    lib_report['libNumResDefraged'] = len(idx2rmv)

    #1) Finalize the "official" sequence to be saved in FASTA file
    fasta_file = os.path.join(lib_dir, fname + '.fasta')

    seqres_file = os.path.join(src_dir, fname + '.seqres')
    _resnums_new, istart, seqres_seq = [], 0, ''
    if os.path.exists(seqres_file) :
        seqres_obj = molstru.SeqStruct2D(seqres_file)
        seqres_seq = ''.join(seqres_obj.seq)

        # seqres and seqatm must be substring one way or the other
        try :
            if len(seqres_seq) > len(atoms_seq) :
                istart = seqres_seq.index(atoms_seq)
                # resnums_new = list(range(istart+1, istart+1+len(atoms_seq)))
                # iseq2add = [list(range(0,istart)), list(range(istart+len(atoms_seq), len(seqres_seq)))]
            else :
                istart = atoms_seq.index(seqres_seq)
                # resnums_new = list(range(-istart + 1, -istart + 1 + len(atoms_seq)))
                # iseq2rmv = [list(range(0,istart)), list(range(istart+len(seqres_seq), len(atoms_seq)))]

            print('Consistent SEQRES and SEQATM, using SEQRES record as the official sequence')
            lib_report['libSeqSource'] = 'SEQRES'
            seqres_obj.write_sequence_file(fasta=fasta_file, single=True, id=fname)
        except ValueError :
            lib_report['libSeqSource'] = None
            logger.info('Unmatching SEQRES and SEQATM record, using SEQATM...')
            print('SEQRES:', seqres_seq)
            print('SEQATM:', atoms_seq)

    # take the sequence from structure, and dangling ends will be removed
    if lib_report['libSeqSource'] is None :
        print('Use SEQATM record as the offical sequence ...')
        lib_report['libSeqSource'] = 'SEQATM'
        copy2(os.path.join(lib_dir, fname + '.seqatm'), fasta_file)

    #2) check 2D and 3D structure against the fasta sequence
    if lib_report['libSeqSource'] == 'SEQRES' :
        lib_report['libNumResAdded'] = len(seqres_seq) - len(atoms_seq)
        if len(seqres_seq) > len(atoms_seq) : # currently, this means SEQATM is shorter than SEQRES
            atoms.renumber_residues(startnum=istart+1)
            dbn_dict['seq'] = seqres_seq[0:istart] + dbn_dict['seq'] + seqres_seq[istart+len(atoms_seq):len(seqres_seq)]
            dbn_dict['dbn'] = '.'*istart + dbn_dict['dbn'] + '.'*(len(seqres_seq)-len(atoms_seq)-istart)
        elif len(seqres_seq) < len(atoms_seq) :
            atoms = atoms.iselect(iseq=list(range(istart, istart+len(seqres_seq))))
            atoms.renumber_residues(startnum=1)
            dbn_dict['seq'] = dbn_dict['seq'][istart:istart+len(seqres_seq)]
            dbn_dict['dbn'] = dbn_dict['dbn'][istart:istart+len(seqres_seq)]

        if len(seqres_seq) != len(atoms_seq) :
            atoms.write_pdb(pdb_file)
            atoms.write_fasta(seqatm_file)
            with open(dbn_file, 'w') as ofile :
                ofile.writelines('\n'.join(['>'+dbn_dict['title'], dbn_dict['seq'], dbn_dict['dbn']]))
    else :
        lib_report['libNumResAdded'] = 0

    return lib_report

def debug_dangling_ends_in_dbn(dssr_dbn, max_length=7, fbase=None, jsonfile=None):
    """ used for debugging purposes """
    idbn_breaks = misc.str_find_all(dssr_dbn, '&')
    if len(idbn_breaks) == 0:
        logger.info(f'JSON shows no breaks in: {fbase}, good to go!')
        return True

    num_breaks = len(idbn_breaks)
    logger.info(f'Found {num_breaks} breaks in JSON: {jsonfile}')

    ires2rmv = [] # this is the actual residue idx (from 0) after removing &
    # check from the 5' end (idx_breaks[0] gives the length)
    idbn_last = -1 # the idx of last break i dssr_dbn
    for i in range(num_breaks):
        if idbn_breaks[i] - idbn_last - 1 <= max_length:
            # if the fragment is unpaired or only paired to itself
            if not molstru.dbn2ct(dssr_dbn[idbn_last+1:idbn_breaks[i]],check_closure=True):
                logger.warning(f"The 5' dangling end (dbn: {dssr_dbn[idbn_last+1:idbn_breaks[i]]}) is paired with others!")
            ires2rmv += list(range(idbn_last-i+1, idbn_breaks[i]-i))
            idbn_last = idbn_breaks[i]
        else:
            logger.debug(f"Keeping the 5' danging end (dbn: {dssr_dbn[:idbn_breaks[0]]})")
            break

    # check the 3' end
    idbn_last = len(dssr_dbn)
    for i in range(num_breaks-1, i-1, -1):
        if idbn_last - idbn_breaks[i] - 1 <= max_length:
            if not molstru.dbn2ct(dssr_dbn[idbn_breaks[i]+1:idbn_last],check_closure=True):
                logger.warning(f"The 3' dangling end (dbn: {dssr_dbn[idbn_breaks[i]+1:idbn_last]}) is paired with others!")
            ires2rmv += list(range(idbn_breaks[i]-i, idbn_last-i-1))
            idbn_last = idbn_breaks[i]
        else:
            logger.debug(f"Keeping the 3' danging end (dbn: {dssr_dbn[idbn_breaks[-1]:]})")
            break

    return ires2rmv

def remove_dangling_ends_by_dssr(pdbfile, jsonfile=None, save_dir=None, max_danglen=11,
        renumber_residues=True, write_pdb=False, write_dbn=False, write_fasta=False, **DUMMY):
    """ attempt to remove dangling fragments from the ends
    One assumption is that all nucleotides are recognized by DSSR

        max_danglen - the maximum length of dangling ends to be removed

    return True only when a single unbroken chain is left!!!
    """

    pdbid = os.path.splitext(os.path.basename(pdbfile))[0]
    dst_pdbfile = pdbid + '.pdb'
    dst_dbnfile = pdbid + '.dbn'
    dst_fastafile = pdbid + '.fasta'

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  

        dst_pdbfile = os.path.join(save_dir, dst_pdbfile)
        dst_dbnfile = os.path.join(save_dir, dst_dbnfile)
        dst_fastafile = os.path.join(save_dir, dst_fastafile)

    # get DBN from JSON
    if jsonfile is None:
        jsonfile = os.path.splitext(pdbfile)[0] + '.json'

    try:
        with open(jsonfile, 'r') as hfile:
            json_info = json.load(hfile)
    except:
        logger.error(f'Error reading JSON file: {jsonfile} !!!')
        return False

    dssr_dbn = dssr_dbn_from_json(json_info)
    if dssr_dbn is None:
        return False
    idbn_breaks = misc.str_find_all(dssr_dbn, '&')
    num_breaks = len(idbn_breaks)

    if len(idbn_breaks) == 0:
        logger.info(f'JSON shows no breaks for PDB: {pdbfile}, good to go (^_^)')
        if write_pdb and os.path.abspath(pdbfile) != os.path.abspath(dst_pdbfile):
            copy2(pdbfile, dst_pdbfile)
        return True

    logger.info(f'Found {num_breaks} breaks in JSON: {jsonfile}')

    # remove dangling ends
    nseg2rmv = 0
    ires2rmv = [] # this is the actual residue idx (from 0) after removing &

    # check from the 5' end (idx_breaks[0] gives the length)
    idbn_last = -1 # the idx of last break i dssr_dbn
    for i in range(num_breaks):
        if idbn_breaks[i] - idbn_last - 1 <= max_danglen:
            # if the fragment is unpaired or only paired to itself
            if not molstru.dbn2ct(dssr_dbn[idbn_last+1:idbn_breaks[i]], check_closure=True):
                logger.warning(f"The 5' dangling end with dbn: {dssr_dbn[idbn_last+1:idbn_breaks[i]]} is paired with others!")
            ires2rmv += list(range(idbn_last-i+1, idbn_breaks[i]-i))
            idbn_last = idbn_breaks[i]
            nseg2rmv += 1
        else:
            logger.debug(f"Keeping the 5' danging end (dbn: {dssr_dbn[:idbn_breaks[0]]})")
            break

    # check from the 3' end
    idbn_last = len(dssr_dbn)
    for i in range(num_breaks-1, i-1, -1):
        if idbn_last - idbn_breaks[i] - 1 <= max_danglen:
            if not molstru.dbn2ct(dssr_dbn[idbn_breaks[i]+1:idbn_last], check_closure=True):
                logger.warning(f"The 3' dangling end with dbn: {dssr_dbn[idbn_breaks[i]+1:idbn_last]} is paired with others!")
            ires2rmv += list(range(idbn_breaks[i]-i, idbn_last-i-1))
            idbn_last = idbn_breaks[i]
            nseg2rmv += 1
        else:
            logger.debug(f"Keeping the 3' danging end (dbn: {dssr_dbn[idbn_breaks[-1]:]})")
            break

    if len(ires2rmv) == 0 or nseg2rmv < num_breaks:
        logger.warning(f'Failed to remove all breaks, num_breaks: {num_breaks}, num_removed:{nseg2rmv}')
        return False

    # finally remove the residues
    # resnames, resnums, iatom_newres, atom_lines = get_seq_from_atom_lines(fbase)
    atoms_stru = molstru3d.AtomsData(pdbfile)

    num_dbns = len(dssr_dbn) - num_breaks
    if atoms_stru.numResids != num_dbns:
        logger.error(f'Inconsistent residue counts in PDB:{atoms_stru.numResids} and JSON: {num_dbns} for {pdbid} !!!')
        return False
    if atoms_stru.numResids <= len(ires2rmv):
        logger.error(f'All PDB residues are being removed for {pdbid} with PDB: PDB:{atoms_stru.numResids}!!!')
        return False

    atoms_stru.remove_residues(iseq=sorted(ires2rmv))

    if renumber_residues:
        atoms_stru.renumber_residues()

    if write_pdb:
        if os.path.abspath(pdbfile) == os.path.abspath(dst_pdbfile):
            logger.warning(f'Overwriting PDB file: {pdbfile} ...')
        else:
            # copy2(pdbfile, pdb_backup)
            logger.info(f'Saving modified PDB to {dst_pdbfile} ...')
        atoms_stru.write_pdb(dst_pdbfile)
        # atoms_stru.write_fasta(seqatm_file)

    logger.info(f'Successfully removed {nseg2rmv} fragments to get a single chain for {pdbid} !')
    return True

def get_frags_by_dssr(pdbfile, jsonfile, save_dir='', save_prefix='', min_length=18,
        max_nbreaks=100, renumber_residues=True, write_pdb=True, write_fasta=True, 
        write_dbn=True, write_tangle=True, write_dist=True, **DUMMY) :
    """ assume all nucleotides are recognized by DSSR!!! """

    pdbid = Path(pdbfile).stem
    rna_report = {'numSavedFrags': 0}

    if save_dir is None:
        save_dir = ''
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        with open(jsonfile, 'r') as hfile:
            json_info = json.load(hfile)
    except:
        logger.error(f'Error reading JSON file: {jsonfile} !!!')
        return rna_report

    dssr_dbn = dssr_dbn_from_json(json_info)
    if dssr_dbn is None:
        logger.error(f'Error reading DBN from JSON file: {jsonfile} !!!')
        return rna_report
        
    idbn_breaks = misc.str_find_all(dssr_dbn, '&')
    num_breaks = len(idbn_breaks)
    if num_breaks == 0:
        logger.info(f'JSON shows no breaks for PDB: {pdbfile}, good to go (^_^)')
    else:
        logger.info(f'Found {num_breaks} breaks in JSON: {jsonfile}')

        if num_breaks > max_nbreaks:
            logger.info(f'num_breaks: {num_breaks} greater than max_nbreaks: {max_nbreaks}, nothing will be saved!!!')
            return rna_report

    # iseq_newfrag points to the actual sequence without "&"
    pdb_dbn = dssr_dbn.replace('&', "")
    iseq_newfrag = [_idbn - _i for _i, _idbn in enumerate(idbn_breaks)]
    # add the beginning and ending indices
    iseq_newfrag.insert(0, 0)
    iseq_newfrag.append(len(pdb_dbn))

    resnames, resnums, iatom_newres, atom_lines = get_seq_from_atom_lines(pdbfile)
    iatom_newres.append(len(atom_lines))

    for ifrag in range(0, len(iseq_newfrag)-1):
        iseq_start = iseq_newfrag[ifrag]
        iseq_end = iseq_newfrag[ifrag+1] # this actually points to the next seq
        frag_len = iseq_end - iseq_start

        if frag_len < min_length:
            logger.info(f'RCSBid: {pdbid}, nfrags: {num_breaks+1}, fragment: {ifrag+1}, length: {frag_len}, less than {min_length}, skipping...')
            continue
        else:
            logger.info(f'RCSBid: {pdbid}, nfrags: {num_breaks+1}, fragment: {ifrag+1}, length: {frag_len}, saving...')

        rna_report['numSavedFrags'] += 1
        iatom_start = iatom_newres[iseq_start]
        iatom_end = iatom_newres[iseq_end]

        if num_breaks == 0:
            copy2(jsonfile, os.path.join(save_dir, pdbid+'.json'))
            fname_stem = pdbid
        else:
            fname_stem = f'{save_prefix}{pdbid}_frag{ifrag+1}'

        if write_fasta:
            with open(os.path.join(save_dir, fname_stem+'.fasta'), 'w') as hfile:
                hfile.writelines([f'>{fname_stem}\n', ''.join(resnames[iseq_start:iseq_end])])

        if write_pdb:
            with open(os.path.join(save_dir, fname_stem+'.pdb'), 'w') as hfile:
                if renumber_residues :
                    hfile.writelines(renumber_atom_lines(atom_lines[iatom_start:iatom_end],
                        resnum_start=1))
                else :
                    hfile.writelines(atom_lines[iatom_start:iatom_end])

        if write_dbn:
            with open(os.path.join(save_dir, fname_stem+'.dbn'), 'w') as hfile:
                hfile.writelines('\n'.join([
                    f'>{fname_stem}',
                    ''.join(resnames[iseq_start:iseq_end]),
                    pdb_dbn[iseq_start:iseq_end]]))

        if write_tangle:
            save_dssr_torsion_angles(json_info, os.path.join(save_dir, fname_stem+'.tangle'))

        if write_dist:
            pdb_file = os.path.join(save_dir, fname_stem+'.pdb')
            if not os.path.exists(pdb_file):
                print("Cannot find PDB file: {}".format(pdb_file))
            else :
                atoms = molstru3d.AtomsData(pdb_file)
                atoms.calc_res2res_distance(neighbor_only=False, atom='P')
                atoms.write_res2res_dist(os.path.join(save_dir, f'{fname_stem}.dist'))

    return rna_report

def remove_unknown_residues_by_dssr(pdbfile, jsonfile, save_dir='', renumber_residues=True,
        write_torsion=False, write_pdb=True, write_fasta=False,
        return_resnames=False, return_atom_lines=False):
    """ Based on PDB and DSSR JSON output, remove residues not recognized by dssr,
    and save PDB in the new folder (save_dir)

    If all residues are recognized, 
    should re-run DSSR in the new folder!!!

    a SINGLE chain is assumed!

    returns sameSeqPDBDSSR, resnames, atom_lines
    """

    rna_report = {'dssrValidJson': True, 'dssrSameSeqSeqAtm': False}

    pdbid = Path(pdbfile).stem
    if save_dir is None:
        save_dir = ''

    resnames, resnums, iatom_newres, atom_lines = get_seq_from_atom_lines(pdbfile, pdbdir=None)
    if return_resnames: rna_report['resnames'] = resnames
    if return_atom_lines: rna_report['atom_lines'] = atom_lines

    # No real need to read DBN, just for checking consistency
    # dbninfo = read_fasta_dbn(fname, fdir=fdir)
    # if 'seqRaw' not in dbninfo:
    #     logger.warning(f'Invalid DBN file: {fname}.dbn !!!')
    # if len(iseq2resnum) != len(dbninfo['seq']) :
    #     logger.warning(f'Mismatching json and dbn files for {pdbfile} !!!')

    jsondict = dssr_read_json(jsonfile)
    if jsondict is None:
        logger.warning(f'RCSBid: {pdbid}-> Unable to get dssr json: {jsonfile}, returning!!!')
        rna_report['dssrValidJson'] = False
        return rna_report

    if write_torsion: # may contain breaks!!!
        torsionfile = os.path.join(save_dir, pdbid + '.torsion')
        save_dssr_torsion_angles(jsondict, torsionfile)

    # resnum mapping from dssr json output
    if 'nts' not in jsondict.keys() :
        logger.error(f'No "nts" field found in Json file: {jsonfile}!!!')
        return rna_report
    
    if "nt_resnum" not in jsondict['nts'][0].keys():
        logger.error(f'No "nt_resnum" field found in Json file: {jsonfile}!!!')
        return rna_report
    
    dssr_resnums = [_resi.get('nt_resnum', None) for _resi in jsondict['nts']] # this is the ACTUAL resnum in PDB
    dssr_resnums = [_resnum for _resnum in dssr_resnums if _resnum is not None]

    # check if there is any difference
    if len(dssr_resnums) > len(resnums):
        logger.error(f'RCSBid: {pdbid}-> More residues in DSSR ({len(dssr_resnums)}) than in PDB ({len(resnums)})!!!')
        return rna_report
    elif len(dssr_resnums) == len(resnums) :
        logger.info(f"Same numbers of residues in PDB and DSSR for {pdbid}, good to go...")
        rna_report['sameSeqAtmSeqDSSR'] = True
        dssr_resnames = resnames
        dssr_atom_lines = atom_lines
        # copy json and dbn files to save_dir
        copy2(jsonfile, os.path.join(save_dir, pdbid + '.json'))
        dbnfile = Path(jsonfile).with_suffix('.dbn')
        if dbnfile.exists():
            copy2(dbnfile, os.path.join(save_dir, pdbid + '.dbn'))
    else:
        # extract the recognized residues from PDB
        logger.warning(f'RCSBid: {pdbid}-> Fewer residues in DSSR ({len(dssr_resnums)}) than in PDB ({len(resnums)})!!!')
        dssr_resnames, dssr_atom_lines = [], []
        iatom_newres.append(len(atom_lines))
        for _i, resnum in enumerate(dssr_resnums):
            iseq = resnums.index(resnum) # get the index of the new seq
            dssr_resnames.append(resnames[iseq])
            dssr_atom_lines.extend(atom_lines[iatom_newres[iseq]:iatom_newres[iseq+1]])

    if renumber_residues:
        dssr_atom_lines = renumber_atom_lines(dssr_atom_lines, resnum_start=1)

    if write_pdb:
        fname_full = os.path.join(save_dir, pdbid + '.pdb')
        # if os.path.isfile(fname_full):
        #     copy2(fname_full, fname_full+backup_suffix)
        with open(fname_full, 'w') as hfile :
            hfile.writelines(dssr_atom_lines)

    if write_fasta:
        fname_full = os.path.join(save_dir, pdbid + '.fasta')
        # if os.path.isfile(fname_full):
        #     copy2(fname_full, fname_full+backup_suffix)
        with open(fname_full, 'w') as hfile :
            hfile.writelines([f'>{pdbid}\n', dssr_resnames+'\n'])

    if return_resnames: rna_report['resnames'] = dssr_resnames
    if return_atom_lines: rna_report['atom_lines'] = dssr_atom_lines
    return rna_report

def save_dssr_torsion_angles(json_info, save_path='dssr.torsion',
        angles = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi', 'eta', 'theta']):
    """ simply extract and save torsion angles from DSSR json outputs """
    if isinstance(json_info, str) or isinstance(json_info, Path):
        with open(json_info, 'r') as hfile :
            json_info = json.load(hfile)
    if not isinstance(json_info,dict):
        logger.critical(f'Unable to recognize json_info: {json_info} !!!')

    if 'nts' not in json_info.keys() :
        logger.critical('Cannot find "nts" key in json_info!!!')
        return False

    torsion_lines = ['resnum,nt,dbn,' + ','.join(angles) + '\n']

    for resi in json_info['nts']:
        new_line = [f'{resi["nt_resnum"]}', resi["nt_code"], resi["dbn"]]

        for angle in angles:
            new_line.append('1000' if resi[angle] is None else f'{resi[angle]:7.3f}')

        torsion_lines.append(','.join(new_line) + '\n')

    logger.info(f'Saving torsion angles to {save_path} ...')
    with open(save_path, 'w') as iofile:
        iofile.writelines(torsion_lines)

    return True

def check_rna_library(fn_prefix, dist_cutoff=7, libdir='', dbn_suffix='.dbn', fasta_suffix='.fasta') :
    '''check consistency between 1D (fasta), 2D (dbn), and 3D (pdb) data'''

    rna_report = dict()

    # resname_list, resnum_list, iatom_newseq, atom_lines = get_seq_from_atom_lines(fn_prefix, pdbdir=libdir)
    atoms = molstru.AtomsData(pdbfile=fn_prefix+'.pdb', pdbdir=libdir)
    atoms.calc_res2res_distance(neighbor_only=True)
    idx_breaks = [_i+1 for _i, _dist in enumerate(atoms.dist_res2res_closest) if _dist  >= dist_cutoff]
    rna_report['libNumPDBBreaks'] = len(idx_breaks) -1
    rna_report['libIdxPDBBreaks'] = str(idx_breaks[1:])
    atoms_seq = ''.join(atoms.seqResidNames)

    # collect information about atoms 3D structure
    rna_report['libNumAtoms'] = atoms.numAtoms
    rna_report['libNumPAtoms'] = len([_s for _s in atoms.atomLines if _s[12:16].strip() == 'P'])
    rna_report["libNumC3'Atoms"] = len([_s for _s in atoms.atomLines if _s[12:16].strip() == "C3'"])
    rna_report["libNumN3'Atoms"] = len([_s for _s in atoms.atomLines if _s[12:16].strip() == 'N3'])

    rna_report['libNumResidues'] = atoms.numResids
    rna_report['chkAllRNAResidues'] = all([resname in molstru_config.ResNames_RNA for resname in atoms.seqResidNames])
    rna_report['chkContiResNum'] = (atoms.residNums[-1]-atoms.residNums[0]+1) == atoms.numResids
    rna_report['chkNoResInsert'] = all([_s[26] == ' ' for _s in atoms.atomLines])
    if not rna_report['chkNoResInsert'] :
        print("PDB has residue insertion code: {}".format(fn_prefix))
    rna_report['chkNoAtomAltloc'] = all([_s[16] == ' ' for _s in atoms.atomLines])

    # 1) 1D consistency: FASTA vs DBN
    fasta_file = os.path.join(libdir, fn_prefix + fasta_suffix)
    fasta_seq = ''
    if os.path.exists(fasta_file) :
        with open(fasta_file, 'r') as hfile :
            fasta_title = hfile.readline().strip()
            fasta_seq = hfile.readline().strip()

        rna_report['chkSameFastaPDB'] = (fasta_seq == atoms_seq)
        rna_report['chkSameNameTitle'] = (fasta_title[1:].startswith(os.path.basename(fn_prefix)))

    rna_report['chkHasDBN'] = False
    dbn_file =  os.path.join(libdir, fn_prefix + dbn_suffix)
    if os.path.exists(dbn_file) :
        dbninfo = read_fasta_dbn(dbn_file)
        rna_report['libNumDBNBreaks'] = dbninfo['numBreaks']
        rna_report['libNumDBNResidues'] = len(dbninfo['seq'])
        rna_report['chkHasDBN'] = 'dbn' in dbninfo.keys()
        rna_report['chkSameNameTitle'] = rna_report['chkSameNameTitle'] and \
                    (dbninfo['title'].startswith(os.path.basename(fn_prefix)))
        rna_report['chkSameFastaDBN'] = dbninfo['seq'] == fasta_seq
        rna_report['chkSamePDBDBN'] = dbninfo['seq'] == atoms_seq

    jsonfile = os.path.join(libdir, fn_prefix+'.json')
    if os.path.exists(jsonfile) :
        try :
            with open(jsonfile, 'r') as hfile :
                json_info = json.load(hfile)

            if 'nts' in json_info.keys() :
                iseq2resnum = [resi['nt_resnum'] for resi in json_info['nts']]
                rna_report['chkDBNContiRes'] = all([(iseq2resnum[i]-iseq2resnum[i-1]) == 1 \
                        for i in range(1, len(iseq2resnum))])
        except :
            pass
    return rna_report
