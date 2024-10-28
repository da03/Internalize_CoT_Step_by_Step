import tracemalloc
import time
import functools
import logging

logger = logging.getLogger(__name__)

def trace_performance(func):
    """A decorator that logs the execution time and memory usage of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Start tracking memory allocations
        tracemalloc.start()
        start_time = time.time()  # Record the start time

        result = func(*args, **kwargs)  # Call the function

        current, peak = tracemalloc.get_traced_memory()  # Get memory usage
        tracemalloc.stop()  # Stop tracking memory

        # Log the time and memory usage
        logger.info(f"[{func.__name__}] Current memory usage {current / 1e6:.3f}MB; Peak: {peak / 1e6:.3f}MB")
        logger.info(f"[{func.__name__}] Time elapsed: {time.time() - start_time:.2f}s")

        return result

    return wrapper
