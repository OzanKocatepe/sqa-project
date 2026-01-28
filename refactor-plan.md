# SSH Model Refactoring - Class Design Template

## USER INPUT FLOW DIAGRAM
```
User provides parameters once at creation:
    ↓
SSHEnsemble.__init__(t1, t2, decayConstant, drivingAmplitude, drivingFreq)
    ↓
SSHEnsemble.add_momentum(k)  ← User adds momentum points
    ↓
SSHEnsemble.run_all(tauAxis, initialConditions, numT, steadyStateCutoff)  ← User provides simulation parameters
    ↓
[System runs, stores all results internally]
    ↓
User accesses results via properties (no further input needed)
```

---

## LEGEND
```
PUBLIC METHODS:      MethodName()
PROTECTED METHODS:   _MethodName()
PRIVATE METHODS:     __MethodName()

PUBLIC ATTRIBUTES:   attributeName
PROTECTED ATTRIBUTES: _attributeName
PRIVATE ATTRIBUTES:   __attributeName

[USER INPUT] = User must provide this value
[COMPUTED]   = System computes this value
[STORED]     = System stores this value from another class
```

---

## 1. PARAMETERS MODULE

### Class: SSHParameters
**Purpose:** Store and validate all physical parameters for a single momentum point
**User Input Point:** Created by SSHEnsemble when user calls add_momentum()

```
ATTRIBUTES:
  PUBLIC:
    k                    : float           [USER INPUT via SSHEnsemble.add_momentum()]
    t1                   : float           [USER INPUT via SSHEnsemble.__init__()]
    t2                   : float           [USER INPUT via SSHEnsemble.__init__()]
    decayConstant        : float           [USER INPUT via SSHEnsemble.__init__()]
    drivingAmplitude     : float           [USER INPUT via SSHEnsemble.__init__()]
    drivingFreq          : float           [USER INPUT via SSHEnsemble.__init__()]

METHODS:
  PUBLIC:
    __init__(k, t1, t2, decayConstant, drivingAmplitude, drivingFreq)
        └─> Initializes all parameters
    
    validate() -> bool
        └─> Validates parameter ranges and physical constraints
    
    to_dict() -> dict
        └─> Returns parameters as dictionary
    
    __repr__() -> str
        └─> String representation for debugging
```

---

## 2. PHYSICS MODULE

### Class: SSHHamiltonian
**Purpose:** Calculate Hamiltonian and eigensystem
**User Input Point:** None (uses SSHParameters)

```
ATTRIBUTES:
  PRIVATE:
    __params             : SSHParameters   [STORED from constructor]
  
  PROTECTED:
    _eigenvalues         : np.ndarray      [COMPUTED, cached]
    _eigenvectors        : np.ndarray      [COMPUTED, cached]
    _eigenvalues_cached  : bool            [COMPUTED]

METHODS:
  PUBLIC:
    __init__(params: SSHParameters)
        └─> Stores parameters
    
    hamiltonian(tau: float = 0) -> np.ndarray
        └─> Returns 2x2 Hamiltonian matrix (time-dependent if tau given)
    
    eigenvalues() -> np.ndarray
        └─> Returns eigenvalues (cached after first call)
    
    eigenvectors() -> np.ndarray
        └─> Returns eigenvectors (cached after first call)
  
  PROTECTED:
    _compute_eigensystem() -> tuple[np.ndarray, np.ndarray]
        └─> Computes and caches eigenvalues and eigenvectors
    
    _clear_cache() -> None
        └─> Clears cached eigenvalues/eigenvectors
```

### Class: SSHDissipator
**Purpose:** Handle dissipation and decay terms
**User Input Point:** None (uses SSHParameters)

```
ATTRIBUTES:
  PRIVATE:
    __params             : SSHParameters   [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(params: SSHParameters)
        └─> Stores parameters
    
    lindblad_operators() -> list[np.ndarray]
        └─> Returns list of Lindblad operators
    
    decay_rate() -> float
        └─> Returns the decay rate (gamma_-)
```

---

## 3. SOLVER MODULE

### Class: ODESolver (Abstract Base Class)
**Purpose:** Define interface for all solvers
**User Input Point:** Not instantiated directly

```
METHODS:
  PUBLIC (ABSTRACT):
    solve(tauAxis: np.ndarray, initialConditions: np.ndarray, **kwargs) -> np.ndarray
        └─> Must be implemented by subclasses
```

### Class: SingleTimeSolver
**Purpose:** Solve single-time correlation ODEs
**User Input Point:** Called by SSHModel.solve()

```
ATTRIBUTES:
  PRIVATE:
    __hamiltonian        : SSHHamiltonian  [STORED from constructor]
    __dissipator         : SSHDissipator   [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(hamiltonian: SSHHamiltonian, dissipator: SSHDissipator)
        └─> Stores physics objects
    
    solve(tauAxis: np.ndarray, initialConditions: np.ndarray, 
          drivingTerm: Callable = None) -> np.ndarray
        └─> Returns solution array of shape (3, len(tauAxis))
        [USER INPUT: tauAxis, initialConditions via SSHEnsemble.run_all()]
  
  PROTECTED:
    _ode_system(tau: float, y: np.ndarray, drivingTerm: Callable) -> np.ndarray
        └─> Defines the ODE system dy/dtau = f(tau, y)
```

### Class: DoubleTimeSolver
**Purpose:** Solve double-time correlation ODEs
**User Input Point:** Called by SSHModel.solve()

```
ATTRIBUTES:
  PRIVATE:
    __hamiltonian        : SSHHamiltonian  [STORED from constructor]
    __dissipator         : SSHDissipator   [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(hamiltonian: SSHHamiltonian, dissipator: SSHDissipator)
        └─> Stores physics objects
    
    solve(tauAxis: np.ndarray, tAxis: np.ndarray,
          singleTimeSolution: np.ndarray) -> np.ndarray
        └─> Returns solution array of shape (3, 3, len(tAxis), len(tauAxis))
        [USER INPUT: tauAxis via SSHEnsemble.run_all()]
        [COMPUTED: tAxis from steady state analysis]
        [STORED: singleTimeSolution from SingleTimeSolver]
  
  PROTECTED:
    _ode_system(tau: float, y: np.ndarray, t: float, 
                singleTimeSolution: np.ndarray) -> np.ndarray
        └─> Defines the ODE system for double-time correlations
```

---

## 4. ANALYSIS MODULE

### Class: FourierAnalyzer
**Purpose:** All Fourier transform operations
**User Input Point:** Called by SSHModel internally

```
ATTRIBUTES:
  PRIVATE:
    __params             : SSHParameters   [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(params: SSHParameters)
        └─> Stores parameters
    
    calculate_coefficients(solution: np.ndarray, tauAxis: np.ndarray,
                          steadyStateCutoff: float, numPeriods: int = 10,
                          n: int = None) -> np.ndarray
        └─> Returns Fourier coefficients of shape (3, 2*n+1)
        [STORED: solution, tauAxis from solver]
        [USER INPUT: steadyStateCutoff via SSHEnsemble.run_all()]
    
    evaluate_expansion(coefficients: np.ndarray, freq: float = None) -> Callable
        └─> Returns function that evaluates Fourier expansion at given times
    
    frequency_axis(tauAxis: np.ndarray) -> np.ndarray
        └─> Calculates frequency axis for FFT
  
  PROTECTED:
    _calculate_max_n(tauAxis: np.ndarray) -> int
        └─> Determines maximum n based on Nyquist frequency
```

### Class: CurrentCalculator
**Purpose:** Calculate current operator in time and frequency domains
**User Input Point:** Called by SSHModel.calculate_current()

```
ATTRIBUTES:
  PRIVATE:
    __params             : SSHParameters   [STORED from constructor]
    __fourier_analyzer   : FourierAnalyzer [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(params: SSHParameters, fourier_analyzer: FourierAnalyzer)
        └─> Stores parameters and analyzer
    
    calculate(correlationData: CorrelationData, 
              steadyStateCutoff: float) -> CurrentData
        └─> Returns CurrentData object with time and frequency domain results
        [STORED: correlationData from SSHModel]
        [USER INPUT: steadyStateCutoff via SSHEnsemble.run_all()]
  
  PROTECTED:
    _calculate_current_coefficients(n: int) -> np.ndarray
        └─> Analytical current coefficients using Bessel functions
    
    _calculate_time_domain(expectationCoeff: np.ndarray, 
                          currentCoeff: np.ndarray) -> np.ndarray
        └─> Computes current in time domain
    
    _calculate_frequency_domain(timeDomain: np.ndarray, 
                               tauAxis: np.ndarray) -> tuple[np.ndarray, np.ndarray]
        └─> Computes FFT of current operator
```

---

## 5. DATA MODULE

### Class: CorrelationData
**Purpose:** Store all correlation function results
**User Input Point:** Created internally by SSHModel.solve()

```
ATTRIBUTES:
  PUBLIC:
    singleTime           : np.ndarray      [STORED from SingleTimeSolver]
                           Shape: (3, len(tauAxis))
    
    doubleTime           : np.ndarray      [STORED from DoubleTimeSolver]
                           Shape: (3, 3, len(tAxis), len(tauAxis))
    
    tauAxisSec           : np.ndarray      [USER INPUT via SSHEnsemble.run_all()]
    
    tauAxisDim           : np.ndarray      [COMPUTED from tauAxisSec]
    
    tAxisSec             : np.ndarray      [COMPUTED from steady state]
    
    tAxisDim             : np.ndarray      [COMPUTED from tAxisSec]
    
    parameters           : SSHParameters   [STORED]
    
    fourierCoefficients  : np.ndarray      [COMPUTED by FourierAnalyzer]
                           Shape: (3, 2*n+1)

METHODS:
  PUBLIC:
    __init__(singleTime, doubleTime, tauAxisSec, tAxisSec, parameters, 
             fourierCoefficients = None)
        └─> Stores all data
    
    get_single_time(operator_index: int) -> np.ndarray
        └─> Returns single-time correlation for operator (0=-, 1=+, 2=z)
    
    get_double_time(i: int, j: int) -> np.ndarray
        └─> Returns double-time correlation <σ_i(t) σ_j(t+τ)>
```

### Class: CurrentData
**Purpose:** Store current operator results
**User Input Point:** Created internally by CurrentCalculator

```
ATTRIBUTES:
  PUBLIC:
    timeDomain           : np.ndarray      [COMPUTED by CurrentCalculator]
    
    freqDomain           : np.ndarray      [COMPUTED by CurrentCalculator]
    
    freqAxis             : np.ndarray      [COMPUTED from tauAxis]
    
    parameters           : SSHParameters   [STORED]

METHODS:
  PUBLIC:
    __init__(timeDomain, freqDomain, freqAxis, parameters)
        └─> Stores all data
```

---

## 6. MODEL MODULE

### Class: SSHModel
**Purpose:** Orchestrate all components for a single momentum point
**User Input Point:** Created by SSHEnsemble.add_momentum()

```
ATTRIBUTES:
  PUBLIC:
    params               : SSHParameters   [STORED from constructor]
  
  PROTECTED:
    _hamiltonian         : SSHHamiltonian  [CREATED in __init__]
    _dissipator          : SSHDissipator   [CREATED in __init__]
    _fourier_analyzer    : FourierAnalyzer [CREATED in __init__]
    _correlation_data    : CorrelationData [COMPUTED in solve()]
    _current_data        : CurrentData     [COMPUTED in calculate_current()]

METHODS:
  PUBLIC:
    __init__(parameters: SSHParameters)
        └─> Creates physics and analysis objects
    
    solve(tauAxis: np.ndarray, initialConditions: np.ndarray,
          numT: int = 5, steadyStateCutoff: float = 25,
          drivingTerm: Callable = None) -> None
        └─> Orchestrates solving single and double-time correlations
        [USER INPUT: tauAxis, initialConditions, numT, steadyStateCutoff 
                     via SSHEnsemble.run_all()]
    
    calculate_current(steadyStateCutoff: float) -> None
        └─> Computes current using CurrentCalculator
        [USER INPUT: steadyStateCutoff via SSHEnsemble.run_all()]
  
  PROPERTIES (READ-ONLY):
    correlation_data     : CorrelationData
        └─> Access correlation results
    
    current_data         : CurrentData
        └─> Access current results
    
    k                    : float
        └─> Momentum value (from params.k)
  
  PROTECTED:
    _determine_t_axis(steadyStateCutoff: float, numT: int) -> np.ndarray
        └─> Calculates tAxis for steady-state initial conditions
```

### Class: SSHEnsemble
**Purpose:** Manage multiple momentum points, main user interface
**User Input Point:** PRIMARY ENTRY POINT FOR USERS

```
ATTRIBUTES:
  PROTECTED:
    _base_params         : dict            [USER INPUT via __init__]
                           {t1, t2, decayConstant, drivingAmplitude, drivingFreq}
    
    _models              : dict[float, SSHModel]  [CREATED by add_momentum()]

METHODS:
  PUBLIC:
    __init__(t1: float, t2: float, decayConstant: float,
             drivingAmplitude: float, drivingFreq: float)
        └─> PRIMARY USER INPUT POINT for physical parameters
        └─> Stores base parameters for all momentum points
    
    add_momentum(k: float | np.ndarray) -> None
        └─> USER INPUT POINT for momentum values
        └─> Creates SSHModel for each k
    
    run_all(tauAxis: np.ndarray, initialConditions: np.ndarray,
            numT: int = 5, steadyStateCutoff: float = 25,
            drivingTerm: Callable = None, debug: bool = False) -> None
        └─> USER INPUT POINT for simulation parameters
        └─> Runs solve() and calculate_current() for all models
    
    get_model(k: float) -> SSHModel
        └─> Returns specific momentum model for detailed access
  
  PROPERTIES (READ-ONLY):
    models               : dict[float, SSHModel]
        └─> Access all models
    
    momentums            : np.ndarray
        └─> Array of all momentum values
    
    tauAxisSec           : np.ndarray
        └─> Tau axis (returns from first model)
    
    tauAxisDim           : np.ndarray
        └─> Dimensionless tau axis
    
    tAxisSec             : np.ndarray
        └─> T axis
    
    tAxisDim             : np.ndarray
        └─> Dimensionless t axis
    
    freqAxis             : np.ndarray
        └─> Frequency axis for Fourier transforms
```

### Class: TotalCurrentCalculator
**Purpose:** Aggregate current across all momentum points
**User Input Point:** Called by visualization or analysis code

```
ATTRIBUTES:
  PRIVATE:
    __ensemble           : SSHEnsemble     [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(ensemble: SSHEnsemble)
        └─> Stores ensemble reference
    
    calculate() -> tuple[np.ndarray, np.ndarray]
        └─> Returns (total_time_domain, total_freq_domain)
        └─> Sums current across all momentum points
```

---

## 7. VISUALIZATION MODULE

### Class: PlotStyler
**Purpose:** Common styling and formatting for all plots
**User Input Point:** None (internal defaults)

```
ATTRIBUTES:
  PUBLIC:
    t_label              : str             [CONSTANT] = r"$t \gamma_-$"
    tau_label            : str             [CONSTANT] = r"$\tau \gamma_-$"
    plotting_functions   : list[Callable]  [CONSTANT] = [abs, real, imag]

METHODS:
  PUBLIC:
    __init__()
        └─> Initializes styling constants
    
    format_title(params: SSHParameters, k: float = None) -> str
        └─> Generates consistent plot title with parameters
    
    format_operator_label(operator_index: int) -> str
        └─> Returns LaTeX label for operator (σ_-, σ_+, σ_z)
```

### Class: SingleTimeCorrelationPlotter
**Purpose:** Plot single-time correlations only
**User Input Point:** User calls via SSHVisualizer

```
ATTRIBUTES:
  PRIVATE:
    __styler             : PlotStyler      [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(styler: PlotStyler)
        └─> Stores styler
    
    plot(model: SSHModel, overplot_fourier: bool = False) -> None
        └─> Creates 3x3 subplot of single-time correlations
        [USER PROVIDES: model (via SSHVisualizer)]
  
  PROTECTED:
    _create_subplot_grid() -> tuple[Figure, np.ndarray]
        └─> Creates figure with 3x3 subplots
    
    _plot_correlation(ax, data: np.ndarray, plotting_func: Callable) -> None
        └─> Plots single correlation on given axis
```

### Class: DoubleTimeCorrelationPlotter
**Purpose:** Plot double-time correlations only
**User Input Point:** User calls via SSHVisualizer

```
ATTRIBUTES:
  PRIVATE:
    __styler             : PlotStyler      [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(styler: PlotStyler)
        └─> Stores styler
    
    plot(model: SSHModel, slice: list[tuple[int]] = None,
         num_tau_points: int = None, save_figs: bool = False,
         subtract_uncorrelated: bool = False) -> None
        └─> Creates 3D plots of double-time correlations
        [USER PROVIDES: model, options via SSHVisualizer]
  
  PROTECTED:
    _create_tau_mask(total_points: int, num_points: int) -> np.ndarray
        └─> Creates mask for downsampling tau axis
    
    _plot_correlation_3d(ax, t_data: np.ndarray, tau_data: np.ndarray,
                        z_data: np.ndarray) -> None
        └─> Plots single 3D correlation surface
```

### Class: CurrentPlotter
**Purpose:** Plot current operator
**User Input Point:** User calls via SSHVisualizer

```
ATTRIBUTES:
  PRIVATE:
    __styler             : PlotStyler      [STORED from constructor]

METHODS:
  PUBLIC:
    __init__(styler: PlotStyler)
        └─> Stores styler
    
    plot_single_momentum(model: SSHModel) -> None
        └─> Plots current for single k value
        [USER PROVIDES: model via SSHVisualizer]
    
    plot_total_current(ensemble: SSHEnsemble) -> None
        └─> Plots summed current across all k
        [USER PROVIDES: ensemble via SSHVisualizer]
  
  PROTECTED:
    _plot_time_domain(current_data: CurrentData) -> None
        └─> Creates time-domain current plot
    
    _plot_frequency_domain(current_data: CurrentData) -> None
        └─> Creates frequency-domain current plot
```

### Class: SSHVisualizer
**Purpose:** Unified interface for all visualization (Facade pattern)
**User Input Point:** Main visualization entry point for users

```
ATTRIBUTES:
  PUBLIC:
    styler                      : PlotStyler                    [CREATED in __init__]
    single_time_plotter         : SingleTimeCorrelationPlotter [CREATED in __init__]
    double_time_plotter         : DoubleTimeCorrelationPlotter [CREATED in __init__]
    current_plotter             : CurrentPlotter               [CREATED in __init__]

METHODS:
  PUBLIC:
    __init__()
        └─> Creates all plotter objects
    
    plot_single_time(model: SSHModel, **kwargs) -> None
        └─> USER ENTRY POINT for single-time plots
        └─> Delegates to single_time_plotter
    
    plot_double_time(model: SSHModel, **kwargs) -> None
        └─> USER ENTRY POINT for double-time plots
        └─> Delegates to double_time_plotter
    
    plot_current(model: SSHModel, **kwargs) -> None
        └─> USER ENTRY POINT for single k current plots
        └─> Delegates to current_plotter.plot_single_momentum()
    
    plot_total_current(ensemble: SSHEnsemble, **kwargs) -> None
        └─> USER ENTRY POINT for total current plots
        └─> Delegates to current_plotter.plot_total_current()
```

---

## COMPLETE USER WORKFLOW

```
STEP 1: Create ensemble with physical parameters
────────────────────────────────────────────────
ensemble = SSHEnsemble(
    t1=2.0,                    # [USER INPUT]
    t2=1.0,                    # [USER INPUT]
    decayConstant=0.1,         # [USER INPUT]
    drivingAmplitude=0.2,      # [USER INPUT]
    drivingFreq=2/3.01         # [USER INPUT]
)
└─> SSHEnsemble stores _base_params dict

STEP 2: Add momentum points
────────────────────────────
ensemble.add_momentum(np.pi/4)  # [USER INPUT]
ensemble.add_momentum([0, np.pi/2, np.pi])  # [USER INPUT - can add multiple]
└─> For each k:
    └─> Creates SSHParameters(k=k, **_base_params)
    └─> Creates SSHModel(parameters)
    └─> Stores in _models[k]

STEP 3: Run simulations
───────────────────────
tauAxis = np.linspace(0, 300, 200000)  # [USER INPUT]
initialConditions = np.array([-0.5, -0.5, 0])  # [USER INPUT]

ensemble.run_all(
    tauAxis=tauAxis,           # [USER INPUT - ONLY TIME GIVEN]
    initialConditions=initialConditions,  # [USER INPUT - ONLY TIME GIVEN]
    numT=5,                    # [USER INPUT - ONLY TIME GIVEN]
    steadyStateCutoff=25       # [USER INPUT - ONLY TIME GIVEN]
)
└─> For each SSHModel in _models:
    └─> model.solve(tauAxis, initialConditions, numT, steadyStateCutoff)
        └─> Creates SingleTimeSolver
        └─> single_solution = solver.solve(tauAxis, initialConditions)
        └─> Determines tAxis from steadyStateCutoff and numT
        └─> Creates DoubleTimeSolver
        └─> double_solution = solver.solve(tauAxis, tAxis, single_solution)
        └─> Creates FourierAnalyzer
        └─> coefficients = analyzer.calculate_coefficients(...)
        └─> Stores CorrelationData(_correlation_data)
    └─> model.calculate_current(steadyStateCutoff)
        └─> Creates CurrentCalculator
        └─> current_data = calculator.calculate(_correlation_data, steadyStateCutoff)
        └─> Stores CurrentData(_current_data)

STEP 4: Visualize results (no more user input needed!)
──────────────────────────────────────────────────────
viz = SSHVisualizer()
model = ensemble.get_model(np.pi/4)

viz.plot_single_time(model, overplot_fourier=True)
viz.plot_double_time(model, subtract_uncorrelated=True)
viz.plot_current(model)
viz.plot_total_current(ensemble)

└─> All data already stored in models
└─> No need to pass tauAxis, parameters, etc. again
└─> Visualization just accesses properties
```

---

## ATTRIBUTE OWNERSHIP TABLE

| Attribute            | Storage Location           | Input Method                | Input Frequency |
|----------------------|----------------------------|-----------------------------|-----------------|
| t1, t2               | SSHParameters              | SSHEnsemble.__init__()      | Once at start   |
| decayConstant        | SSHParameters              | SSHEnsemble.__init__()      | Once at start   |
| drivingAmplitude     | SSHParameters              | SSHEnsemble.__init__()      | Once at start   |
| drivingFreq          | SSHParameters              | SSHEnsemble.__init__()      | Once at start   |
| k                    | SSHParameters              | SSHEnsemble.add_momentum()  | Once per k      |
| tauAxis              | CorrelationData            | SSHEnsemble.run_all()       | Once at run     |
| initialConditions    | Passed to solve()          | SSHEnsemble.run_all()       | Once at run     |
| numT                 | Passed to solve()          | SSHEnsemble.run_all()       | Once at run     |
| steadyStateCutoff    | Passed to solve()          | SSHEnsemble.run_all()       | Once at run     |
| tAxis                | CorrelationData (computed) | Auto-computed internally    | Never (auto)    |
| singleTimeSolution   | CorrelationData (computed) | Auto-computed internally    | Never (auto)    |
| doubleTimeSolution   | CorrelationData (computed) | Auto-computed internally    | Never (auto)    |
| currentTime          | CurrentData (computed)     | Auto-computed internally    | Never (auto)    |
| currentFreq          | CurrentData (computed)     | Auto-computed internally    | Never (auto)    |
| freqAxis             | CurrentData (computed)     | Auto-computed internally    | Never (auto)    |
| fourierCoefficients  | CorrelationData (computed) | Auto-computed internally    | Never (auto)    |

---

## KEY DESIGN PRINCIPLES

1. **Single Source of Truth**: Each piece of user input is provided exactly once
   - Physical parameters → SSHEnsemble.__init__()
   - Momentum points → SSHEnsemble.add_momentum()
   - Simulation parameters → SSHEnsemble.run_all()

2. **Progressive Enhancement**: User builds up the system step-by-step
   - Create ensemble → Add momentums → Run simulations → Visualize
   
3. **No Redundant Storage**: Attributes stored at the most appropriate level
   - Shared parameters (t1, t2, etc.) → SSHParameters
   - Computed results → Data classes (CorrelationData, CurrentData)
   - Analysis tools → Analyzer classes

4. **Clean Access**: Results accessed via properties, never by passing data around
   - model.correlation_data.singleTime
   - model.current_data.timeDomain
   - ensemble.tauAxisDim

5. **Encapsulation**: Internal details hidden
   - User never sees SingleTimeSolver directly
   - User never manually creates FourierAnalyzer
   - Protected/private methods do the heavy lifting