# Problem Statement

The Jy current is not zero in the non-trivial phase $\Delta = 3$, even though it should be.

The error is systematic, a very accurate sine wave with amplitude ~0.01.

# Log

## Incorrect Formulae in Code

At the very least, the Fourier series for $h_x(t), h_y, h_z(t), H_m(t), H_p(t), H_z(t)$ match their Fourier series in the code. Tested it at $(k_x, k_y) = (\pi / 4, -\pi / 8)$ and $(k_x, k_y) = (\pi / 4.01, -\pi / 8.07)$.

Also, based on eyeballed measurements it definitely seems like $H_m(t) = H_p(t)^*$, so at least those formulae seem to be correct?

The single-time correlations seem right, with $\sigma_-$ and $\sigma_-$ oscillating around zero, and being conjugates of each other. However, the $\sigma_z$ doesn't oscillate around -1, but rather 'bounces' off of it? It doesn't even bounce evenly, it seems to vary in amplitude. 'Bouncing' might be the wrong term, the most negative it goes is -1. Will double-check if this agrees with numerical simulation. However, its worrying since the difference in amplitude is like 1e-4, and thats also the order of amplitude of the oscillation, so idk if that can explain a current of 0.1 amplitude appearing in the y-direction.

The $\sigma_z$ term seems weird, but it matches the numerical solution, so unless the ODE is wrong I don't think there's a problem there. I should compare with Alessandria's code.

I've looked at every formula in the Hamiltonian class and they match my derivations. Need to check whether my derivations are correct.

Seems like we're using the right number of momentum points.

Replacing my current functions with Alessandria's (switching the sign of the sin components), without changing everything else (other than sometimes the sign of the driving amplitude to match hers) actually makes the code worse, even if I set the BZ offsets back to 0. Switching between our derivations must be slightly more complicated, maybe requiring changing the sigma ODE too?

## Rho Singularities

There are singularities at $\{ (x, y) : x, y \in \{-\pi, 0, \pi \} \}$. These points have $\rho = 0$, so the assumption is that because $\rho$ gets so small, at opposite momentums the value of $\rho$ undergoes rounding errors, meaning $1 / \rho$ doesn't match at opposite momentums, and hence the Jy current terms don't cancel.

## Inaccuracies in Conjugacy

We found that the differences between $\sigma_-$ and $\sigma_+^*$, and $j_{y, -}$ and $j_{y, +}^*$ are on the order of 1e-18, so the conjugacy doesn't have nearly enough magnitude of systematicness of the error to explain the problem.