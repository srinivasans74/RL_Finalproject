#!/bin/sh
#python main.py --env-name HalfCheetah-v4 --cost-limit 0.04
#python main.py --env-name HalfCheetah-v4 --cost-limit $1
# python main.py --env-name HalfCheetah-v4 --cost-limit 100.0


start=0.1
end=0.9
increment=0.1

# Loop from start to end, incrementing by increment
for i in $(seq $start $increment $end); do
    echo "Iteration: $i"
    python main.py --env-name HalfCheetah-v4 --cost-limit $i
done