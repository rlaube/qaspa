@echo off

set seed=0
set outdir=csqa/permutation_seed%seed%.npy

call activate qaspa
call python generate_perm_vector.py --outdir %outdir% --seed %seed%
call conda deactivate
