from dataclasses import dataclass, field
import time
import os
import threading

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

    # We use a singleton instance for every process.
    _instance: SSHProfiler = None

    def __new__(cls):
        """
        When creating an instance, returns the singleton instance
        if it already exists.
        """

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Tells the initialiser that this is a newly constructed instance.
            cls._instance.initialised = False
        return cls._instance
    
    def __init__(self):
        """Initialises the singleton instance if it has just been created."""

        if not self.initialised:
            self.__records: list[ProfileRecord] = []
            # Only allows one thread at a time to write to the records.
            self.__lock = threading.Lock()
            self.__currentMomentum: float = None
            self.__initialised = True

    def SetMomentum(self, momentum: float) -> None:
        """
        Allows the SSH process to set the momentum, since each process controls one SSH model,
        which has a fixed momentum.
        """