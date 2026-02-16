from dataclasses import dataclass, field
import time
import os
import pandas as pd

# Global variable determining whether the process is enabled.
PROFILING_ENABLED = True

@dataclass
class ProfileRecord:
    """A single profiling record. Stores all relevant information."""
    functionName: str
    momentum: float | None = None
    elapsedTime: float = 0.0
    numCalls: int = 0
    # processID: int = field(default_factory = os.getpid)

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

    def __getitem__(self, key: str) -> ProfileRecord:
        """
        Allows the instance to be indexed by a function name.
        Returns the corresponding record, creating it if it doesn't
        yet exist.

        Parameters
        ----------
        key : str
            The key, which should be the function name.

        Returns
        -------
        ProfileRecord
            The corresponding profile record.
        """

        for record in self.__records:
            # If we have found this function in the record, it should be the only one
            # with that function.
            if record.functionName == key:
                return record
        
        # If we didn't find it in the records, creates a new record for that function name.
        newRecord = ProfileRecord(
            functionName = key,
            momentum = self.__k,
        )

        # Appends to itself and returns.
        self.append(newRecord)
        return newRecord

    def append(self, record: ProfileRecord) -> None:
        """
        Appends a profile record to the internal record.
        
        Parameters
        ----------
        record : ProfileRecord
            The record to append to the list.
        """

        self.__records.append(record)

    @staticmethod
    def profile(f):
        """
        Measures the execution time of a function and stores it to a dataset
        for analysis.
        """

        # Wrapping function takes self as an argument, since we only call
        # profiler on instance methods.
        # If we need functionality for profiling class/static methods, we will
        # need another wrapper.
        def wrap(self, *args, **kwargs):
            # Calculates the elapsed time for the function to be called.
            initialTime = time.perf_counter()
            result = f(self, *args, **kwargs)
            elapsedTime = time.perf_counter() - initialTime

            # Gets the instance of the profiler within this SSH object,
            # and adds to the elapsed time for this function.
            self.profiler[f.__name__].elapsedTime += elapsedTime
            self.profiler[f.__name__].numCalls += 1

            # Returns the result of the original function call.
            return result

        return wrap
    
    def ExportToDataframe(self) -> pd.DataFrame:
        """
        Exports the records to a pandas dataframe.
        
        Returns
        -------
        Dataframe
            The dataframe containing the data stored in the record.
        """

        return pd.DataFrame(self.__records)
    
    @property
    def k(self) -> float:
        return self.__k