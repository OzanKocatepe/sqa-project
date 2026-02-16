from dataclasses import dataclass, field
import time
import os

# Global variable determining whether the process is enabled.
PROFILING_ENABLED = True

@dataclass
class ProfileRecord:
    """A single profiling record. Stores all relevant information."""
    functionName: str
    momentum: float | None = None
    elapsedTime: float = 0.0
    timestamp: float = field(default_factory = time.time)
    processID: int = field(default_factory = os.getpid)

class SSHProfiler:
    """
    A profiler for the SSH class. Used to optimise the simulation.
    """
 
    def __init__(self, k: float):
        """
        Initialises the singleton instance if it has just been created.
        
        k : float
            The momentum of the SSH object that this profiler is
            attached to.
        """

        self.__k = k
        self.__records: list[ProfileRecord] = []

    def profile(f):
        """
        Measures the execution time of a function and stores it to a dataset
        for analysis.
        """

        def wrap(*args, **kwargs):
            initialTime = time.perf_counter()
            result = f(*args, **kwargs)
            elapsedTime = time.perf_counter() - initialTime
            print(f"{elapsedTime:.2f}s")
            return result

        return wrap