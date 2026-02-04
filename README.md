# SQA Project
This repo contains the code I used to model resonance fluorescence in a variety of systems. Currently, we can model a qubit in a waveguide (following [Kocabas et. al., 2012](https://arxiv.org/abs/1111.7315))
and are currently in the process of modelling a one-dimensional Su–Schrieffer–Heeger (SSH) model with an infinite bulk.

# SSH Model
The relevant code for the 1-dimensional SSH model is stored within _SSHModel/_ as a package. We assume a chain with entirely real hopping amplitudes $t_1, t_2$, an infinite bulk (and
hence periodic boundary conditions), and a classical driving field $A(t) = A_0 \sin(2\pi \Omega t)$. In this section we will describe each package at a high-level (more detailed documentation
can be found within the code) and some of the relevant theory. We always work in the eigenbasis of the unperturbed Hamiltonian.

## Fourier
The Fourier class is a convenient class for storing a fourier series. A new instance can be constructed using either the exponential Fourier coefficients $c_{-n}, \dots, c_n$, or the $x$ and $y$ data of
some periodic function, where it will then calculate the coefficients. Throughout the code, we always use exponentials whose frequencies are integer harmonics of the driving angular frequency $2\pi\Omega$.

The _Evaluate()_ function will take in an array of points and evaluate the fourier series at those points.

## CorrelationData
This is another data class that stores all of the single-time correlations $\langle \sigma_i (\tau) \rangle$, and double-time correlations $\langle \sigma_i(t) \sigma_j(t + \tau) \rangle$ for $i, j \in \{ +, -, z \}$.
It also stores the $\tau$ and $t$ axes that we will use in both seconds and dimensionless units.

## CurrentData
Another data class that stores the single-time current correlation in the time-domain, $\langle j(\tau) \rangle$, and the frequency domain, $\langle \tilde{j} (\omega) \rangle$. It also
contains the double-time correlations $\int dt\, \langle j(t) j(t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$. Further derived current products can also be found here.
