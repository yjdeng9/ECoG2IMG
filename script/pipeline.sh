#!/usr/bin/env bash

## step 1: train TaMAE to get the latent representation
#python trainTaMAE_2d.py subj1
#python trainTaMAE_2d.py subj3
#python trainTaMAE_2d.py subj4
#python trainTaMAE_2d.py subj6
#python trainTaMAE_2d.py subj7
#python trainTaMAE_2d.py subj8
#python trainTaMAE_2d.py subj9
#python trainTaMAE_2d.py subj10
#python trainTaMAE_2d.py subj12
#python trainTaMAE_2d.py subj17
#python trainTaMAE_2d.py subj18

## step 2: train Aligner to align the latent representation
#python trainAligner_cross.py

## step 3: reconstruct Visual Image
#python reconECoG.py subj1
#python reconECoG.py subj3
#python reconECoG.py subj4
#python reconECoG.py subj6
#python reconECoG.py subj7
#python reconECoG.py subj8
#python reconECoG.py subj9
#python reconECoG.py subj10
#python reconECoG.py subj12
#python reconECoG.py subj17
#python reconECoG.py subj18
