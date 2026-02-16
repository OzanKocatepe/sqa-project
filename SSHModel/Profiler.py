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

    @staticmethod
    def profile(f):
        """
        Measures the execution time of a function and stores it to a dataset
        for analysis.
        """

        # Wrapping function takes self as an argument.
        # Not sure if this means it would break for class methods, but as
        # far as I can recall we don't have any of those.
        def wrap(self, *args, **kwargs):
            # Calculates the elapsed time for the function to be called.
            initialTime = time.perf_counter()
            result = f(self, *args, **kwargs)
            elapsedTime = time.perf_counter() - initialTime

            # Stores the data within the internal list.
            self.__records.append(
                ProfileRecord(
                    functionName = f.__name__,
                    momentum = self.__k,
                    elapsedTime = elapsedTime
                )
            )

            # Returns the result of the original function call.
            return result

        return wrap