chern_insulator/src/
│
├── config/                        # unchanged
│   └── paths.py
│
├── data/                        # inputs to the simulation
│   ├── parameters.py              # EnsembleParameters, ModelParameters
│   └── axes.py                    # AxisData
│   ├── model_data.py              # Model Data (per k-point), to be BZ-averaged and post-processed
│   └── ensemble_data.py           # Calculations using the BZ-averaged and post-processed results.
│
├── physics/                       # physical operators and topology; pure functions
│   └── fourier.py                 # Fourier class (currently in data/)
│   ├── hamiltonian.py             # hx/hy/hz, energy, Ax, lattice_basis, Fourier coeffs
│   ├── band_basis.py              # BandBasis dataclass + ALL projection/rotation fns
│   │                              #   (merges data/band_basis.py + operators/band_basis_projector.py)
│   ├── current_operators.py       # ParamagneticCurrentX/Y, DiamagneticCurrentXX/YY
│   └── topology.py                # berry_curvature, chern_number
│
├── observables/               # computing physical observables from solved dynamics
│   ├── correlation_solver.py               # equation_of_motion() — the ODE RHS only
│   ├── model_calculations.py      # paramagnetic, diamagnetic, total current  [per k-point]
│   └── ensemble_calculations.py   # mode populations, g2(0), squeezing  [post-BZ-avg only]
│
├── simulation/                    # orchestration only — no physics logic here
│   ├── model.py                   # single k-point: calls dynamics/ then observables/
│   └── ensemble.py                # BZ sampling, multiprocessing, averaging
│
└── main.py

# Results

[ModelData]
First Order Correlations
Paramagnetic Current
Diamagnetic Current
Total Current
DC Population Variance (Weak Laser)
Generalised Noise Tensor (Weak Laser) (t Averaged)
Second Order Correlations
Second Order Connected Current

Current Fourier Coefficients [intermediate product]

[EnsembleData]
Spectral Noise Tensor
Semiclassical Mode Population
Second Order Connected Current (t Averaged)
Second Order Correlation Function
DC Population Variance
Generalised Noise Tensor
Squeezing (Weak Laser)
Squeezing
Angular Momentum