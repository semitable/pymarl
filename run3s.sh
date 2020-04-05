#!/bin/bash
for i in {0..2}
do
   python src/main.py --config=$1 --env-config=sc2 with env_args.map_name=$2 seed=$i &
   echo "Running with $1 and $2 for seed=$i"
done

