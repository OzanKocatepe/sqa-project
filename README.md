# SQA Project
This repo contains the code I used to model HHG in a variety of systems. Currently, we can model a qubit in a waveguide (following [Kocabas et. al., 2012](https://arxiv.org/abs/1111.7315))
and a one-dimensional Su–Schrieffer–Heeger (SSH) model with an infinite bulk (though this may contain minor errors).

## Installation

This tutorial will assume you are using the package manager ```uv```. This can be installed using most modern package managers (I believe). You can use other package managers, including just pip, but the same commands will not work to automatically download the required packages.

Navigate to sqa-project/chern_insulator. From here, run

```
uv sync
```

to install all the necessary packages. If prompted, make sure to create a virtual environment. Then, change your environment by running

```
source .venv/bin/activate
```

from within the chern_insulator directory. Now, you can run the code using

```
chern_insulator [--args]
```

from any directory, if you are in the virtual environment. The required arguments can be found by running the script with the ```-h``` flag.

For benchmarking purposes, run the command

```
time chern_insulator -k 3 -t 3 -c 1
```

This runs the script using a 3x3 sampling grid on the FBZ, with 3 steady-state points chosen as initial conditions for the second-order correlation functions, and runs the code on a single core for consistency.