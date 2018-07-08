#!/bin/bash

## Script to measure the coverage of the test suite (via doctest).
## Launch it using ./coverage
## and open the html files under the folder htmlcov/
## Skip xpure.py as it is not really part of the pipeline
for i in spectractor/*.py
do
    coverage run -a --source=spectractor $i
done

for i in spectractor/extractor/*.py
do
    coverage run -a --source=spectractor $i
done


