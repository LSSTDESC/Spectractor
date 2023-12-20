#!/bin/bash

## Script to measure the coverage of the test suite (via doctest).
## Launch it using ./coverage
## and open the html files under the folder htmlcov/
echo Data files:
ls tests/data/*.fits
for i in spectractor/*.py
do
    echo Testing $i...
    coverage run -m pytest -a --source=spectractor $i
done

for i in spectractor/extractor/*.py
do
    echo Testing $i...
    coverage run -m pytest -a --source=spectractor $i
done

for i in spectractor/simulation/*.py
do
    echo Testing $i...
    coverage run -m pytest -a --source=spectractor $i
done

for i in spectractor/fit/*.py
do
    echo Testing $i...
    coverage run -m pytest -a --source=spectractor $i
done

