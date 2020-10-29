# Learning from Trajectories via Subgoal Discovery

## Overview
This package is a PyTorch implementation of the paper [Learning from Trajectories via Subgoal Discovery](https://intra.ece.ucr.edu/~supaul/Webpage_files/subgoals_neurips_2019.pdf), by [Sujoy Paul](https://intra.ece.ucr.edu/~supaul/
), [Jeroen van Baar](http://www.merl.com/people/jeroen) and [Amit K Roy-Chowdhury](https://vcg.engr.ucr.edu/amit), published at NeurIPS 2019. 

## Usage

python main.py --path-to-trajectories <path/to/trajectories/>

Here is a [link](https://drive.google.com/file/d/12BJIAAESs-Iy33jHDS4Nde1ZgwbgoIZ-/view?usp=sharing) to the AntMaze trajectories. The outputs can also be visualized with vis.py as follows.

python vis.py --path-to-trajectories <path/to/trajectories> --pretrained-ckpt <checkpointname>


