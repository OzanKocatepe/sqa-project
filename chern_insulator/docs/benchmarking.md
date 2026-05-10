# Benchmarking

As of writing, we can solve for the first- and second-order correlation functions for both the pauli operators and
the current operators. However, this takes time. It is worth writing how each optimisation has impacted the code.

All benchmarking is done on a model with a single momentum point, with numT = 5 and tauMax = 20, on a single core.

## Naive Approach: 1m 30s

We solve the system using the regular ODE function for each initial time $t$ and left operator $\sigma_i (t)$ separately.

## DOP853: 1m 30s -> 30s (3x)

Simply changing the ODE solver method from RK45 to DOP853.

## Batching: 30s -> 10s (3x)

Instead of solving for each left operator separately, we batch the ODE. We solve for all left- and right- operators for an initial time $t$ by putting the columns (corresponding to each left-operator) together into a matrix, and solving that using the same ODE. This does require some reshaping to work with solve_ivp, but its a noticeable speedup.

# Changing rtol and atol: 10s -> 3s (~3x)

The tolerances were a bit overkill. Changed rtol from 1e-11 -> 1e-3 and atol from 1e-12 -> 1e-6. The outputs have technically changed according to git, but I cannot tell the difference at all so I think thats well within tolerance (also the scales we are working with means generally those tolerances sound good).