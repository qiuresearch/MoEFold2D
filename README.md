# MoEFold2D
A Mixture of Expert Approach to RNA Secondary Structure Prediction based on a Leave-One-Cluster-Out (LOCO) approach

## Contents
1. src folder
   - contains all python codes for training, evaluation, and prediction
2. models folder
   - contains saved SeqFold2D LOCO models. For each model, you can find 
      - args.json, the configuration file
      - minets_paddle.py, the source code at the time of model creation
      - net.state, the model state dictionary
      - opt.state, the optimizer state dictionary
3. examples folder
   - contains example fasta input files
4. run_moefold2d.sh - the main script

   
## Install
The simplest way is to create a new anaconda environment using the environment_[cpu|gpu].yml file by running:

`conda env create -f environment.yml`

This will create conda environment "paddle26".


## Usage
run_moefold2d.sh should be the only script needed to run the code. 

### Help
run_moefold2d.sh can be run without argument to get the following help:

```
Usage: run_moefold2d.sh action data_files [cmOptions]

Arguments:
    action          : only predict is supported at this time
    data_files      : pkl file(s) for train, fasta file(s) for predict, folder for brew_dbn/ct/bpseq
    -upsample    [] : use models trained with upsampling (default: True)
    -model       [] : run a single model from the following (default: all models will be run):
                          metafam3.metafam2d.l4c64.validobj.16S-rRNA                              
                          metafam3.metafam2d.l4c64.validobj.23S-rRNA                              
                          metafam3.metafam2d.l4c64.validobj.5S-rRNA                               
                          metafam3.metafam2d.l4c64.validobj.gpI-intron                            
                          metafam3.metafam2d.l4c64.validobj.RNaseP                                
                          metafam3.metafam2d.l4c64.validobj.SRP                                   
                          metafam3.metafam2d.l4c64.validobj.TERC                                  
                          metafam3.metafam2d.l4c64.validobj.tmRNA                                 
                          metafam3.metafam2d.l4c64.validobj.tRNA                                  
                          metafam3.metafam2d.l4c64.validobj.all                                   

    -cmdArgs        : all other options are passed to fly_paddle.py as-is

    SPACE in folder/file names will very likely BREAK the code!!!
```

### Predict
We will use the 100 sequences in the examples/stralign_nr80_100Seqs.fasta file:

```
run_moefold2d.sh predict examples/metafam2d_nr80_test_p1.fasta 
```

### Outputs
1. Intermediate files. A folder "loco_outputs" will be created under current directory. Predictions from each model will be saved into a subfolder and a dbn file and pairing probability matrix will be saved for each sequence.

2. MoE predictions will be saved into a folder named "moe_outputs" containing a dbn file for each sequence. 

3. The predicted RNA type is saved as the "cluster_id" column in a csv file: moefold_eps-auto.csv.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)