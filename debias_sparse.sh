#! /bin/bash

export ALPHA=0
python debias_sparse.py 2>&1 | tee debias_sparse.txt
