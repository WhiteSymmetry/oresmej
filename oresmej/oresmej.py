# oresmej.py
"""
A module for generating Oresme numbers (harmonic series partial sums)
"""

from enum import Enum, auto
from functools import partial, lru_cache
from fractions import Fraction
import jax
import jax.numpy as jnp
import logging
import math
import numpy as np
import os
import time
from typing import Any, Dict, List, Union, Generator, Tuple, Optional
import oresme
import oresmen

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
    """Bernoulli numbers (cached): Bernoulli sayılarını hesaplar (önbellekli olabilir)."""
    if n == 0:
        return 1.0
    elif n == 1:
        return -0.5
    elif n % 2 != 0:
        return 0.0
    else:
        from scipy.special import bernoulli
        return bernoulli(n)[n]

# -----------------------------
# Performance Analysis
# -----------------------------
def benchmark_harmonic(compute_funcs: Dict[str, callable], n: int, runs: int = 10) -> dict:
    """
    Benchmark given compute functions.
    Verilen hesaplama fonksiyonlarını kıyaslar.
    """
    results = {}
    for name, func in compute_funcs.items():
        # warm-up (JAX varsa block_until_ready)
        try:
            func(10).block_until_ready()
        except Exception:
            pass
        start = time.perf_counter()
        for _ in range(runs):
            func(n)
        elapsed = time.perf_counter() - start
        results[name] = elapsed / runs
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

def is_in_hilbert(sequence: Union[List[float], np.ndarray, Generator[float, None, None]], 
                 max_terms: int = 10000, 
                 tolerance: float = 1e-6) -> bool:
    """
    Test whether a sequence belongs to ℓ² (Hilbert space).
    Bir dizinin ℓ² (Hilbert) uzayında olup olmadığını test eder.
    Determines if a given sequence belongs to the Hilbert space ℓ².
    A sequence {a_n} is in ℓ² (Hilbert space) if the sum of the squares of its terms is finite:
        Σ |a_n|² < ∞
    This function computes the partial sum of squared terms up to `max_terms` and checks
    whether the sum converges within a given tolerance (i.e., the increments become negligible).
    Parameters
    ----------
    sequence : list, np.ndarray, or generator
        The input sequence to test (e.g., [1, 1/2, 1/3, ...]).
    max_terms : int, optional
        Maximum number of terms to consider for convergence check. Default is 10,000.
    tolerance : float, optional
        The threshold for determining convergence. If the increment in cumulative sum
        falls below this value for consecutive steps, the series is considered convergent.
        Default is 1e-6.
    Returns
    -------
    bool
        True if the sequence is likely in ℓ² (sum of squares converges), False otherwise.
    Examples
    --------
    >>> from oresmen import harmonic_numbers_numba, is_in_hilbert
    >>> import numpy as np
    # Harmonic terms: a_n = 1/n → sum(1/n²) converges → in Hilbert space
    >>> n = 1000
    >>> harmonic_terms = 1 / np.arange(1, n+1)
    >>> is_in_hilbert(harmonic_terms)
    True
    # Constant terms: a_n = 1 → sum(1²) = ∞ → not in Hilbert space
    >>> constant_terms = np.ones(1000)
    >>> is_in_hilbert(constant_terms)
    False
    Notes
    -----
    - This is a numerical approximation. True mathematical convergence may require
      analytical proof, but this function provides a practical check for common sequences.
    - Sequences like 1/n, 1/n^(0.6), log(n)/n are tested implicitly via their decay rate.
    """

    # Convert generator to list if needed
    if isinstance(sequence, Generator):
        sequence = list(sequence)

    arr = np.array(sequence, dtype=float)

    if arr.size == 0:
        return True

    if not np.all(np.isfinite(arr)):
        return False

    n_terms = min(len(arr), max_terms)
    test_seq = arr[:n_terms]

    squares = test_seq ** 2
    cumsum = np.cumsum(squares)
    total_sum = cumsum[-1]

    if not np.isfinite(total_sum):
        return False

    # p‑series heuristic – eşik 100
    if n_terms > 100 and np.all(test_seq[100:] > 0):
        log_terms = np.log(test_seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, n_terms))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            if alpha > 0.5:
                return True
            elif 0 < alpha <= 0.5:
                return False
            elif alpha > 10:   # üstel sönüm
                return True
        except Exception:
            pass

    # kuyruk katkısı
    if n_terms > 1000:
        last_contribution = squares[-1000:]
        if np.sum(last_contribution) < tolerance:
            return True

    # oran testi (üstel sönüm)
    if n_terms > 100:
        ratios = np.abs(test_seq[1:100] / (test_seq[:99] + 1e-12))
        if np.mean(ratios) < 0.95:
            return True

    # Hiçbir yakınsama belirtisi yoksa ℓ²'de değildir
    return False

# -----------------------------
# Utility Functions / Yardımcı Fonksiyonlar
# -----------------------------
def harmonic_sequence(n_terms: int, start: int = 1) -> np.ndarray:
    """Generate harmonic sequence terms: a_n = 1/n / Harmonik dizi terimlerini üretir: a_n = 1/n"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / indices


def p_series(p: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Generate p-series: a_n = 1/n^p / p-serisi üretir: a_n = 1/n^p"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / (indices ** p)


def geometric_sequence(ratio: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Generate geometric sequence: a_n = ratio^n / Geometrik dizi üretir: a_n = ratio^n"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    exponents = np.arange(start, start + n_terms, dtype=float)
    return ratio ** exponents

# -----------------------------
# Analysis utilities / Analiz araçları
# -----------------------------
def analyze_sequence(
    sequence: Union[List[float], np.ndarray],
    name: str = "Sequence / Dizi",
    n_display: int = 5
) -> dict:
    """Detailed analysis of a sequence / Bir dizinin detaylı analizi"""
    seq = np.array(sequence, dtype=float)
    squares = seq ** 2
    cumsum = np.cumsum(squares)
    results = {
        'name': name,
        'first_terms': seq[:n_display].tolist(),
        'n_terms': len(seq),
        'sum_of_squares': cumsum[-1] if np.isfinite(cumsum[-1]) else np.inf,
        'in_hilbert': is_in_hilbert(seq),
        'max_term': float(np.max(np.abs(seq))),
        'decay_rate': None
    }
    if len(seq) > 100 and np.all(seq[100:] > 0):
        log_terms = np.log(seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, len(seq)))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            results['decay_rate'] = alpha
            results['decay_description'] = f"~ 1/n^{alpha:.2f}"
        except Exception:
            pass
    return results


def compare_sequences(sequences: dict, n_test: int = 5000) -> None:
    """Compare multiple sequences / Birden fazla diziyi karşılaştırır"""
    results = []
    for name, seq in sequences.items():
        if len(seq) < n_test:
            # Auto-extend / Otomatik uzat
            if "n/2" in name or "Oresme" in name:
                indices = np.arange(1, n_test + 1)
                seq = indices / (2.0 ** indices)
            elif "1/n" in name and "1/n²" not in name and "1/n³" not in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / indices
            elif "1/n²" in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / (indices ** 2)
            elif "1/√n" in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / np.sqrt(indices)
            elif "e⁻ⁿ" in name or "exp" in name:
                indices = np.arange(1, n_test + 1)
                seq = np.exp(-indices)
        test_seq = seq[:n_test]
        squares_sum = np.sum(test_seq ** 2)
        in_hilbert = is_in_hilbert(test_seq)
        results.append({
            "Sequence / Dizi": name,
            "First 5 terms / İlk 5 terim": str(test_seq[:5].tolist())[:60],
            "∑ a_n²": f"{squares_sum:.6f}" if np.isfinite(squares_sum) else "∞",
            "In ℓ²? / ℓ²'de mi?": "✓ Yes / Evet" if in_hilbert else "✗ No / Hayır"
        })
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))
    except ImportError:
        for row in results:
            print(row)


# -----------------------------
# Visualization / Görselleştirme
# -----------------------------
def plot_comparative_performance(max_n=50000, step=5000, runs=10):
    """Comparative performance plot / Karşılaştırmalı performans grafiği"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting / Grafik için matplotlib gereklidir")
        return

    n_values = list(range(5000, max_n + 1, step))
    results = {'python': [], 'numpy': [], 'approx': []}

    for n in n_values:
        # Python
        py_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = harmonic_number(n)
            py_times.append(time.perf_counter() - t0)

        # NumPy
        np_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = harmonic_numbers_jax(n)
            np_times.append(time.perf_counter() - t0)

        # Approx
        approx_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = harmonic_number_approx(n)
            approx_times.append(time.perf_counter() - t0)

        results['python'].append(np.mean(py_times) * 1000)
        results['numpy'].append(np.mean(np_times) * 1000)
        results['approx'].append(np.mean(approx_times) * 1000)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, results['python'], 'b-o', label='Pure Python')
    plt.plot(n_values, results['numpy'], 'r-s', label='NumPy')
    plt.plot(n_values, results['approx'], 'g-^', label='Approximate')
    plt.title('Performance Comparison (oresme.py)')
    plt.xlabel('n')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_benchmarks(n: int = 100_000, runs: int = 5):
    """
    Performans karşılaştırması: oresme (pure), oresmen (Numba), oresmej (JAX)
    Her modülün kendi içindeki en hızlı yöntemleri sözlük ile benchmark_harmonic'e gönderir.
    """
    # JAX'ı CPU moduna zorla (adil karşılaştırma)
    import oresmej
    oresmej.enable_gpu(False)

    print(f"Karşılaştırmalı Performans Testi (n={n:,}, runs={runs})")
    print("=" * 70)

    # --- oresme (Pure) ---
    import oresme
    start = time.perf_counter()
    bench_pure = oresme.benchmark_harmonic({
        "pure_python": lambda n: oresme.harmonic_number(n),
        "numpy":       lambda n: oresme.harmonic_numbers_numpy(n),
        "approx":      lambda n: oresme.harmonic_number_approx(n)
    }, n, runs)
    elapsed_pure = time.perf_counter() - start

    # --- oresmen (Numba) ---
    import oresmen
    start = time.perf_counter()
    bench_numba = oresmen.benchmark_harmonic({
        "python_jit":     lambda n: oresmen.harmonic_number(n),
        "numba_vectorized": lambda n: oresmen.harmonic_number_numba(n),
        "approx":         lambda n: oresmen.harmonic_number_approx(n)
    }, n, runs)
    elapsed_numba = time.perf_counter() - start

    # --- oresmej (JAX) ---
    start = time.perf_counter()
    bench_jax = oresmej.benchmark_harmonic({
        "pure_python": lambda n: oresmej.harmonic_number(n),
        "jax":         lambda n: oresmej.harmonic_number_jax(n).block_until_ready(),
        "approx":      lambda n: oresmej.harmonic_number_approx(n)
    }, n, runs)
    elapsed_jax = time.perf_counter() - start

    # Sonuçları birleştir ve tablo yap
    all_results = {}
    for method, t in bench_pure.items():
        all_results[("oresme", method)] = t
    for method, t in bench_numba.items():
        all_results[("oresmen", method)] = t
    for method, t in bench_jax.items():
        all_results[("oresmej", method)] = t

    # Tablo yazdır
    print(f"{'Modül':<10} {'Yöntem':<25} {'Süre (ms)':>10} {'Hız (runs/s)':>12}")
    print("-" * 70)
    for (mod, method), t in sorted(all_results.items()):
        ms = t * 1000
        rps = 1.0 / t if t > 0 else float('inf')
        print(f"{mod:<10} {method:<25} {ms:10.4f} {rps:12.2f}")

    print("-" * 70)
    print(f"Toplam test süreleri: oresme={elapsed_pure:.4f}s, "
          f"oresmen={elapsed_numba:.4f}s, oresmej={elapsed_jax:.4f}s")

    # En hızlı yöntemi bul
    fastest = min(all_results.items(), key=lambda x: x[1])
    print(f"\nEn hızlı yöntem: {fastest[0][0]} -> {fastest[0][1]} "
          f"({fastest[1]*1000:.4f} ms)")

def _run_tests(verbose: bool = True) -> bool:
    """
    Dahili test fonksiyonu. Tüm alt fonksiyonları çağırarak temel doğrulamaları yapar.
    Başarı durumunda True döner.
    """
    tests_passed = 0
    tests_failed = 0

    def check(condition, msg):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
            if verbose:
                print(f"  ✓ {msg}")
        else:
            tests_failed += 1
            if verbose:
                print(f"  ✗ {msg}")

    print("Oresme (Pure) Module Tests / Modül Testleri")
    print("=" * 60)

    # 1. Oresme sequence
    seq = oresme_sequence(5)
    check(len(seq) == 5, "oresme_sequence(5) length")
    check(abs(seq[0] - 0.5) < 1e-9, "oresme_sequence first term")
    check(abs(seq[4] - 5/32) < 1e-9, "oresme_sequence 5th term")

    # 2. Fractional harmonic numbers
    h_frac = harmonic_numbers(5)
    check(len(h_frac) == 5, "harmonic_numbers(5) length")
    check(h_frac[0] == Fraction(1,1), "H1 = 1")
    check(h_frac[4] == Fraction(137, 60), "H5 = 137/60")

    # 3. Float harmonic number
    h5 = harmonic_number(5)
    check(abs(h5 - 2.283333333333333) < 1e-6, "harmonic_number(5)")

    # 4. NumPy harmonic numbers
    h_arr = harmonic_numbers_jax(5)
    check(len(h_arr) == 5, "harmonic_numbers_jax(5) length")
    check(abs(h_arr[4] - 2.283333333333333) < 1e-6, "NumPy H5 value")

    # 5. Generator
    gen_vals = list(harmonic_generator_jax(5))
    check(len(gen_vals) == 5, "harmonic_generator_jax(5) length")
    check(abs(gen_vals[4] - 2.283333333333333) < 1e-6, "Generator H5 value")

    # 6. Approximations
    h100_exact = harmonic_number(100)
    h100_approx = harmonic_number_approx(100)
    err = abs(h100_exact - h100_approx) / h100_exact
    check(err < 1e-4, f"Euler-Mascheroni approx error < 1e-4 (actual {err:.2e})")

    h100_mac = harmonic_number_approx(100, method=ApproximationMethod.EULER_MACLAURIN, k=4)
    err_mac = abs(h100_exact - h100_mac) / h100_exact
    check(err_mac < 1e-6, f"Euler-Maclaurin(4) error < 1e-6 (actual {err_mac:.2e})")

    # 7. Vectorized approx
    n_vals = np.array([10, 100, 1000])
    approx_vec = harmonic_sum_approx(n_vals)
    check(len(approx_vec) == 3, "harmonic_sum_approx vector length")
    check(abs(approx_vec[1] - h100_exact) / h100_exact < 1e-4, "Vector approx H100 error")

    # 8. Bernoulli numbers
    B2 = bernoulli_number(2)
    check(abs(B2 - 1/6) < 1e-9, "B2 = 1/6")

    # 9. Hilbert space tests
    seq_1n = 1 / np.arange(1, 1000)
    check(is_in_hilbert(seq_1n) == True, "1/n in ℓ² (True)")
    seq_slow = 1 / np.sqrt(np.arange(1, 1000))
    check(is_in_hilbert(seq_slow) == False, "1/√n not in ℓ² (False)")
    seq_oresme = np.array([i / (2**i) for i in range(1, 500)])
    check(is_in_hilbert(seq_oresme) == True, "n/2^n in ℓ² (True)")
    check(is_in_hilbert(np.ones(1000)) == False, "Constant 1 not in ℓ² (False)")

    # 10. Sequence generators
    hseq = harmonic_sequence(5)
    check(len(hseq) == 5 and abs(hseq[4] - 0.2) < 1e-9, "harmonic_sequence(5)")
    pseq = p_series(2, 5)
    check(len(pseq) == 5 and abs(pseq[4] - 1/25) < 1e-9, "p_series(2,5)")
    gseq = geometric_sequence(0.5, 5)
    check(len(gseq) == 5 and abs(gseq[4] - 0.5**5) < 1e-9, "geometric_sequence(0.5,5)")

    # 11. Analysis and comparison
    try:
        analysis = analyze_sequence(seq_oresme, name="Oresme")
        check(isinstance(analysis, dict), "analyze_sequence returns dict")
        compare_sequences({"1/n": seq_1n, "n/2ⁿ": seq_oresme}, n_test=500)
    except Exception as e:
        check(False, f"Analysis/comparison raised {e}")

    # 12. Convergence analysis
    try:
        conv = harmonic_convergence_analysis(np.array([10, 100, 1000]))
        check('exact_sums' in conv, "harmonic_convergence_analysis keys")
    except Exception as e:
        check(False, f"Convergence analysis raised {e}")

    # 13. Benchmark
    try:
        bench = benchmark_harmonic({
            "python": lambda n: harmonic_number(n),
            "numpy": lambda n: harmonic_numbers_jax(n)
        }, n=100, runs=3)
        check("python" in bench, "benchmark_harmonic keys")
    except Exception as e:
        check(False, f"Benchmark raised {e}")

    print("-" * 60)
    print(f"Tests: {tests_passed} passed, {tests_failed} failed / {tests_passed} başarılı, {tests_failed} başarısız")
    return tests_failed == 0

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

    # Performance test (sözlük ile)
    n_test = 100000
    logger.info("Performance Test (n=%d):", n_test)
    bench_results = benchmark_harmonic({
        "python": lambda n: harmonic_number(n),
        "jax": lambda n: harmonic_number_jax(n).block_until_ready(),
        "approx": lambda n: harmonic_number_approx(n)
    }, n_test, runs=10)
    for method, time_taken in bench_results.items():
        logger.info("%15s: %.6f s/run", method, time_taken)

    # Comparison
    logger.info("Exact/Approximate Value Comparison (H_100):")
    comparison = compare_with_approximation(100)
    for key, value in comparison.items():
        logger.info("%20s: %.10f", key, value)


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
            # Sözlük ile plot_comparative_performance çağrısı
            plot_comparative_performance({
                "python": lambda n: harmonic_number(n),
                "jax": lambda n: harmonic_number_jax(n).block_until_ready(),
                "approx": lambda n: harmonic_number_approx(n)
            }, max_n=50000, step=5000, runs=10)
        else:
            print(f"oresmej {__version__} başarıyla yüklendi")
            main()

    _cli()
