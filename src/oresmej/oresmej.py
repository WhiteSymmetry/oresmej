# oresmej.py
"""
A module for generating Oresme numbers (harmonic series partial sums)
"""
import os
import jax
import jax.numpy as jnp
from functools import partial, lru_cache
from fractions import Fraction
import math
from typing import List, Union, Generator, Tuple, Optional
import time
import logging
from enum import Enum, auto

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('harmonic')
logger.propagate = False  # Prevent duplicate logs

# Add handler only once
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Filter JAX backend messages
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# -----------------------------
# GPU Configuration (Optional)
# -----------------------------

def enable_gpu(enable: bool = True):
    """
    Enable/disable GPU usage
    
    Args:
        enable: If True, attempts to use GPU. If False, forces CPU usage.
    """
    if enable:
        try:
            os.environ["JAX_PLATFORM_NAME"] = "gpu"
            _ = jax.devices("gpu")
            logger.info("GPU backend successfully enabled")
        except RuntimeError:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            logger.warning("GPU not found, using CPU")
    else:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        logger.info("Forcing CPU usage")

# Default to CPU
enable_gpu(False)

# -----------------------------
# Constants and Enums
# -----------------------------

class ApproximationMethod(Enum):
    """Harmonic number approximation methods"""
    EULER_MASCHERONI = auto()
    EULER_MACLAURIN = auto()
    ASYMPTOTIC = auto()

EULER_MASCHERONI = 0.57721566490153286060
EULER_MASCHERONI_FRACTION = Fraction(303847, 562250)

# -----------------------------
# Core Functions
# -----------------------------

def oresme_sequence(n_terms: int, start: int = 1) -> List[float]:
    """Oresme sequence: a_i = i / 2^i"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive")
    return [i / (2 ** i) for i in range(start, start + n_terms)]

@lru_cache(maxsize=128)
def harmonic_numbers(n_terms: int, start_index: int = 1) -> Tuple[Fraction]:
    """Fractional harmonic numbers (cached)"""
    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    if start_index <= 0:
        raise ValueError("start_index must be positive")
        
    sequence = []
    current_sum = Fraction(0)
    for i in range(start_index, start_index + n_terms):
        current_sum += Fraction(1, i)
        sequence.append(current_sum)
    return tuple(sequence)

def harmonic_number(n: int) -> float:
    """n-th harmonic number (float)"""
    if n <= 0:
        raise ValueError("n must be positive")
    return sum(1.0 / k for k in range(1, n + 1))

# -----------------------------
# JAX-Optimized Functions
# -----------------------------

@partial(jax.jit, static_argnums=(0,))
def harmonic_number_jax(n: int) -> float:
    """JIT-compiled harmonic number function"""
    return jnp.sum(1.0 / jnp.arange(1, n + 1))

@partial(jax.jit, static_argnums=(0,))
def harmonic_numbers_jax(n: int) -> jnp.ndarray:
    """JAX-accelerated harmonic numbers"""
    return jnp.cumsum(1.0 / jnp.arange(1, n + 1))

def harmonic_generator_jax(n: int) -> Generator[float, None, None]:
    """JAX-powered harmonic number generator"""
    sums = harmonic_numbers_jax(n)
    for i in range(n):
        yield float(sums[i])

# -----------------------------
# Approximation Functions
# -----------------------------

def harmonic_number_approx(
    n: int, 
    method: ApproximationMethod = ApproximationMethod.EULER_MASCHERONI,
    k: int = 2
) -> float:
    """Approximate harmonic number calculation"""
    if n <= 0:
        raise ValueError("n must be positive")
        
    if method == ApproximationMethod.EULER_MASCHERONI:
        return math.log(n) + EULER_MASCHERONI + 1/(2*n) - 1/(12*n**2)
    elif method == ApproximationMethod.EULER_MACLAURIN:
        result = math.log(n) + EULER_MASCHERONI + 1/(2*n)
        for i in range(1, k+1):
            B = bernoulli_number(2*i)
            term = B / (2*i) * (1/n)**(2*i)
            result -= term
        return result
    elif method == ApproximationMethod.ASYMPTOTIC:
        return math.log(n) + EULER_MASCHERONI + 1/(2*n)
    else:
        raise ValueError("Unknown approximation method")

@lru_cache(maxsize=32)
def bernoulli_number(n: int) -> float:
    """Bernoulli numbers (cached)"""
    if n == 0: return 1.0
    elif n == 1: return -0.5
    elif n % 2 != 0: return 0.0
    else:
        from scipy.special import bernoulli
        return bernoulli(n)[n]

# -----------------------------
# Performance Analysis
# -----------------------------

def benchmark_harmonic(n: int, runs: int = 10) -> dict:
    """Benchmark different calculation methods"""
    results = {}
    
    # Warm-up call
    _ = harmonic_number_jax(10).block_until_ready()
    
    # Pure Python
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number(n)
    results['pure_python'] = (time.perf_counter() - start)/runs
    
    # JAX
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_jax(n).block_until_ready()
    results['jax'] = (time.perf_counter() - start)/runs
    
    # Approximate
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_approx(n)
    results['approximate'] = (time.perf_counter() - start)/runs
    
    return results

def compare_with_approximation(n: int) -> dict:
    """Compare exact and approximate values"""
    exact = harmonic_number(n)
    approx = harmonic_number_approx(n)
    error = abs(exact - approx)
    relative_error = error / exact if exact != 0 else 0
    
    return {
        'exact': exact,
        'approximate': approx,
        'absolute_error': error,
        'relative_error': relative_error,
        'percentage_error': relative_error * 100
    }

# -----------------------------
# Visualization Functions
# -----------------------------

def plot_comparative_performance(max_n=50000, step=5000, runs=10):
    """Comparative performance analysis (first run vs subsequent runs)"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Data preparation
    n_values = list(range(5000, max_n+1, step))
    results = {
        'python': [],
        'jax_first': [],
        'jax_avg': [],
        'approx': []
    }
    
    # Warm-up call
    _ = harmonic_number_jax(100).block_until_ready()
    
    for n in n_values:
        # Python performance
        py_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number(n)
            py_times.append(time.perf_counter() - start)
        
        # JAX performance (first run + average)
        jax_times = []
        start = time.perf_counter()
        _ = harmonic_number_jax(n).block_until_ready()
        first_run = time.perf_counter() - start
        
        for _ in range(runs-1):
            start = time.perf_counter()
            _ = harmonic_number_jax(n).block_until_ready()
            jax_times.append(time.perf_counter() - start)
        
        # Approximate method
        approx_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = harmonic_number_approx(n)
            approx_times.append(time.perf_counter() - start)
        
        # Store results (in milliseconds)
        results['python'].append(np.mean(py_times)*1000)
        results['jax_first'].append(first_run*1000)
        results['jax_avg'].append(np.mean(jax_times)*1000)
        results['approx'].append(np.mean(approx_times)*1000)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # 1. Main Performance Comparison
    plt.subplot(2, 2, 1)
    plt.plot(n_values, results['python'], 'b-o', label='Pure Python')
    plt.plot(n_values, results['jax_first'], 'r--s', label='JAX (first run)')
    plt.plot(n_values, results['jax_avg'], 'r-s', label='JAX (average)')
    plt.plot(n_values, results['approx'], 'g-^', label='Approximate')
    
    plt.title('Performance Comparison of All Methods')
    plt.xlabel('n value')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    
    # 2. JAX First Run vs Average
    plt.subplot(2, 2, 2)
    plt.plot(n_values, results['jax_first'], 'r--s', label='First run (with compilation)')
    plt.plot(n_values, results['jax_avg'], 'r-s', label='Average (subsequent runs)')
    plt.plot(n_values, np.array(results['jax_first'])/np.array(results['jax_avg']), 
             'k-*', label='Speedup ratio (right axis)')
    
    plt.title('JAX: First Run vs Subsequent Runs')
    plt.xlabel('n value')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    
    # Secondary axis
    ax2 = plt.gca().twinx()
    ax2.set_ylabel('Speedup Ratio (times)')
    ax2.plot([], [], 'k-*', label='Speedup ratio')
    ax2.legend(loc='upper right')
    
    # 3. Python vs JAX Average
    plt.subplot(2, 2, 3)
    speedup = np.array(results['python'])/np.array(results['jax_avg'])
    plt.plot(n_values, speedup, 'm-D')
    
    plt.title('JAX Speedup (vs Python)')
    plt.xlabel('n value')
    plt.ylabel('Speedup factor')
    plt.grid(True)
    
    # 4. Zoomed Comparison (n < 15000)
    plt.subplot(2, 2, 4)
    mask = np.array(n_values) <= 15000
    small_n = np.array(n_values)[mask]
    
    plt.plot(small_n, np.array(results['python'])[mask], 'b-o', label='Python')
    plt.plot(small_n, np.array(results['jax_avg'])[mask], 'r-s', label='JAX (avg)')
    plt.plot(small_n, np.array(results['approx'])[mask], 'g-^', label='Approximate')
    
    plt.title('Zoomed View for Small n Values (n ≤ 15,000)')
    plt.xlabel('n value')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Detailed data output
    print("\nDetailed Performance Data (in milliseconds):")
    print(f"{'n':>8} | {'Python':>8} | {'JAX (first)':>10} | {'JAX (avg)':>9} | {'Approx':>8} | {'Speedup':>8}")
    print("-"*70)
    for i, n in enumerate(n_values):
        speedup = results['python'][i]/results['jax_avg'][i]
        print(f"{n:8} | {results['python'][i]:8.3f} | {results['jax_first'][i]:10.3f} | "
              f"{results['jax_avg'][i]:9.3f} | {results['approx'][i]:8.3f} | {speedup:8.2f}x")


# -----------------------------
# Advanced Harmonic Approximations
# -----------------------------

def harmonic_sum_approx(n: Union[float, jnp.ndarray], 
                      method: ApproximationMethod = ApproximationMethod.EULER_MACLAURIN,
                      order: int = 4) -> Union[float, jnp.ndarray]:
    """
    Advanced harmonic series approximation using Euler-Maclaurin formula
    
    Args:
        n: Input value(s) (can be scalar or array)
        method: Approximation method (EULER_MASCHERONI, EULER_MACLAURIN, ASYMPTOTIC)
        order: Order of approximation for Euler-Maclaurin (2, 4, or 6)
    
    Returns:
        Approximate harmonic sum H(n)
    
    Examples:
        >>> harmonic_sum_approx(1e6)
        14.392726722865724
        >>> harmonic_sum_approx(1e6, method=ApproximationMethod.EULER_MASCHERONI)
        14.392726722864
    """
    if isinstance(n, (int, float)) and n <= 0:
        raise ValueError("n must be positive")
    
    gamma = EULER_MASCHERONI
    log_n = jnp.log(n)
    
    if method == ApproximationMethod.EULER_MASCHERONI:
        return gamma + log_n + 1/(2*n)
    
    elif method == ApproximationMethod.ASYMPTOTIC:
        return gamma + log_n
    
    elif method == ApproximationMethod.EULER_MACLAURIN:
        result = gamma + log_n + 1/(2*n)
        
        # 2nd order terms
        if order >= 2:
            result -= 1/(12*n**2)
        
        # 4th order terms
        if order >= 4:
            result += 1/(120*n**4)
            
        # 6th order terms
        if order >= 6:
            result -= 1/(252*n**6)
            
        return result
    
    else:
        raise ValueError("Invalid approximation method")

@partial(jax.jit, static_argnums=(1,2))
def harmonic_sum_approx_jax(n: jnp.ndarray, 
                          method: int = 1,  # 0:EULER_MASCHERONI, 1:EULER_MACLAURIN
                          order: int = 4) -> jnp.ndarray:
    """
    JAX-compatible optimized version of harmonic approximation
    
    Note: Uses integer flags instead of Enum for better JIT compatibility
    """
    gamma = EULER_MASCHERONI
    log_n = jnp.log(n)
    inv_n = 1.0/n
    
    # Base terms
    result = gamma + log_n
    
    if method >= 1:  # EULER_MASCHERONI includes 1/(2n)
        result += 0.5*inv_n
        
        if order >= 2:
            inv_n2 = inv_n*inv_n
            result -= inv_n2/12
            
            if order >= 4:
                inv_n4 = inv_n2*inv_n2
                result += inv_n4/120
                
                if order >= 6:
                    inv_n6 = inv_n4*inv_n2
                    result -= inv_n6/252
    
    return result

# -----------------------------
# Convergence Analysis Utilities
# -----------------------------

def harmonic_convergence_analysis(n_values: jnp.ndarray) -> dict:
    """
    Analyze harmonic series convergence for given values
    
    Args:
        n_values: Array of n values to analyze
    
    Returns:
        Dictionary containing:
        - exact_sums: Exact harmonic sums
        - approx_sums: Approximate sums
        - errors: Absolute errors
        - log_fit: Logarithmic fit coefficients
    """
    exact = harmonic_numbers_jax(n_values[-1])[n_values-1]  # -1 for 0-based indexing
    approx = harmonic_sum_approx_jax(n_values.astype(float))
    
    return {
        'exact_sums': exact,
        'approx_sums': approx,
        'errors': jnp.abs(exact - approx),
        'log_fit': jnp.polyfit(jnp.log(n_values), exact, 1)  # a*ln(n) + b
    }


# -----------------------------
# Main Program
# -----------------------------

def main():
    """Main function"""
    # GPU/CPU configuration
    enable_gpu(False)
    
    # Calculations
    logger.info("Oresme Sequence (first 5 terms): %s", oresme_sequence(5))
    logger.info("Fractional Harmonic Numbers (H1-H3): %s", harmonic_numbers(3))
    logger.info("5th Harmonic Number: %.4f", harmonic_number(5))
    
    # Approximate values
    logger.info("1000th Harmonic Number Approximations:")
    logger.info("Euler-Mascheroni: %.8f", 
               harmonic_number_approx(1000, ApproximationMethod.EULER_MASCHERONI))
    logger.info("Asymptotic: %.8f", 
               harmonic_number_approx(1000, ApproximationMethod.ASYMPTOTIC))
    
    # JAX calculations
    _ = harmonic_number_jax(10).block_until_ready()  # Warm-up
    logger.info("JAX Accelerated (H1-H5): %s", harmonic_numbers_jax(5))
    logger.info("JAX Generator (H1-H3): %s", list(harmonic_generator_jax(3)))
    
    # Performance test
    n_test = 100000
    logger.info("Performance Test (n=%d):", n_test)
    bench_results = benchmark_harmonic(n_test)
    for method, time_taken in bench_results.items():
        logger.info("%15s: %.6f s/run", method, time_taken)
    
    # Comparison
    logger.info("Exact/Approximate Value Comparison (H_100):")
    comparison = compare_with_approximation(100)
    for key, value in comparison.items():
        logger.info("%20s: %.10f", key, value)
        
"""
if __name__ == "__main__":
    main()
    plot_comparative_performance()
"""

__all__ = [
    'oresme_sequence',
    'harmonic_numbers',
    'harmonic_number',
    'harmonic_number_jax',
    'harmonic_numbers_jax',
    'harmonic_generator_jax',
    'harmonic_number_approx',
    'EULER_MASCHERONI',
    'harmonic_sum_approx', 
    'harmonic_sum_approx_jax', 
    'harmonic_convergence_analysis',
]

if __name__ == "__main__":
    def _cli():
        """Konsol arayüzü için ana fonksiyon"""
        from argparse import ArgumentParser
        
        parser = ArgumentParser(description='oresmej: Harmonik sayı ve Oresme dizisi hesaplamaları')
        parser.add_argument('--test', action='store_true', help='Tüm fonksiyonları test et')
        parser.add_argument('--plot', action='store_true', help='Karşılaştırma grafiklerini göster')
        parser.add_argument('-v', '--version', action='version', version=f'oresmej {__version__}')
        args = parser.parse_args()
        
        if args.test:
            from .tests import run_tests  # Test modülünüz varsa
            run_tests()
        elif args.plot:
            plot_comparative_performance()
        else:
            print(f"oresmej {__version__} başarıyla yüklendi")

    _cli()
