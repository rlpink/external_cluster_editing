#!/bin/bash


for graph_pars in "50 50" "100 50" "100 100" "1000 50" "1000 100";
do
	set -- $graph_pars
	for err in "002" "005" "010" "022";
	do
		instance=${1}x${2}_$err
		mkdir data/$instance
		for rep in {1..10};
		do
			mkdir data/$instance/$rep
			seed=$((${1} * ${2} * ${err} * $rep));
    			sbatch -J $instance -o data/$instance/$rep/time_v4.txt experiments_v4.sh $instance.txt $rep $seed
		done;
	done;
done;
