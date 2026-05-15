# Karma Model Phase-Field Simulation

This repository contains a CUDA-accelerated phase-field simulation code for dendritic/cellular solidification based on a Karma-type model. The code evolves the phase field `phi` and solute concentration field `c` under an imposed thermal gradient and pulling velocity.

The simulation is written in Python using NumPy, Matplotlib, and Numba CUDA.

### Requirements

This code requires a CUDA-capable GPU.  
Python packages
numpy
numba
matplotlib

### How to run
python karma.py --G 1e6 --Vs 1e-2

### Output
`dendrite.csv`: interface temperature and pull-back information  
`phi_<iteration>.csv`: phase-field data  
`c_<iteration>.csv`: concentration-field data  
`T_<iteration>.csv`: temperature-field data  
`iteration_<iteration>.png`: visualization of the phase field and concentration field
