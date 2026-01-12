# TO-DO

- Test system with different values of $\tau$, $\tilde \Omega$, and $\omega_R$.
    - [x] Related: Fix the Bloch equation matrix since that form isn't parameterised by $\tau$. Shouldn't matter
    when $\tau$ = 1 but will certainly matter when we change that.

- Figure out why the initial conditions for $\langle \tilde \sigma_z (t) \rangle$ are ~0.5 in magnitude, while the actual initial conditions are -1 (entire value). Oddly enough, the issue is coming from the analytical solution, which should match that initial value.