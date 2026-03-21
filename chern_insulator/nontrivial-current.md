# Problem Statement

The Jy current is not zero in the non-trivial phase $\Delta = 3$, even though it should be.

The error is systematic, a very accurate sine wave with amplitude ~0.01.

# Log

## Rho Singularities

There are singularities at $\{ (x, y) : x, y \in \{-\pi, 0, \pi \} \}$. These points have $\rho = 0$, so the assumption is that because $\rho$ gets so small, at opposite momentums the value of $\rho$ undergoes rounding errors, meaning $1 / \rho$ doesn't match at opposite momentums, and hence the Jy current terms don't cancel.

## Inaccuracies in Conjugacy

We found that the differences between $\sigma_-$ and $\sigma_+^*$, and $j_{y, -}$ and $j_{y, +}^*$ are on the order of 1e-18, so the conjugacy doesn't have nearly enough magnitude of systematicness of the error to explain the problem.