chern_insulator/src/
│
├── config/                        # unchanged
│   └── paths.py
│
├── params/                        # inputs to the simulation
│   ├── parameters.py              # EnsembleParameters, ModelParameters
│   └── axes.py                    # AxisData
│
├── utils/                         # mathematical utilities with no physics content
│   └── fourier.py                 # Fourier class (currently in data/)
│
├── physics/                       # physical operators and topology; pure functions
│   ├── hamiltonian.py             # hx/hy/hz, energy, Ax, lattice_basis, Fourier coeffs
│   ├── band_basis.py              # BandBasis dataclass + ALL projection/rotation fns
│   │                              #   (merges data/band_basis.py + operators/band_basis_projector.py)
│   ├── current_operators.py       # ParamagneticCurrentX/Y, DiamagneticCurrentXX/YY
│   └── topology.py                # berry_curvature, chern_number
│
├── dynamics/                      # solving the equations of motion
│   ├── equations.py               # equation_of_motion() — the ODE RHS only
│   ├── single_time.py             # Fourier-space solver for σ_-, σ_+, σ_z
│   └── double_time.py             # diffrax ODE solver for two-time correlators
│
├── observables/                   # computing physical observables from solved dynamics
│   ├── first_order.py             # paramagnetic, diamagnetic, total current  [per k-point]
│   ├── second_order.py            # connected current, noise tensor, weak-laser noise  [per k-point]
│   └── photon_stats.py            # mode populations, g2(0), squeezing  [post-BZ-avg only]
│
├── results/                       # typed containers for computed output
│   ├── model_results.py           # FirstOrderResults, SecondOrderResults, WeakLaserResults
│   └── ensemble_results.py        # EnsembleResults (BZ-averaged, + post-processed)
│
├── simulation/                    # orchestration only — no physics logic here
│   ├── model.py                   # single k-point: calls dynamics/ then observables/
│   └── ensemble.py                # BZ sampling, multiprocessing, averaging
│
├── visualization/                 # all plotting, split by topic
│   ├── current_plots.py
│   ├── correlation_plots.py
│   └── photon_stats_plots.py
│
└── main.py