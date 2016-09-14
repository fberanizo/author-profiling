#!/bin/bash

for i in `seq 1 10`;
do
    #python test_gender.py > "output/gender.out.$i"
    python test_age.py > "output/age.out.$i"
    #python test_ti.py > "output/ti.out.$i"
    #python test_religiousity.py > "output/religiousity.out.$i"
done