#!/bin/bash

for D1 in 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
do
    echo $D1
    mkdir -p ligands/$D1
    for D2 in `curl -s  https://files.rcsb.org/ligands/$D1/ | sed "s/\"/ /g; s/\// /g" | grep "^<a href=" | awk '{print $3}'`
    do
        echo $D1 $D2
        for ext in model.sdf #ideal.mol2 ideal.sdf model.mol2
        do
            if [ ! -s ligands/$D1/${D2}_$ext ]
            then
                wget https://files.rcsb.org/ligands/$D1/$D2/${D2}_$ext -O ligands/$D1/${D2}_$ext 2> /dev/null
            fi
        done
        if [ ! -s ligands/$D1/${D2}.cif ]
        then
            wget https://files.rcsb.org/ligands/$D1/$D2/${D2}.cif -O ligands/$D1/${D2}.cif 2> /dev/null
        fi
    done
done
