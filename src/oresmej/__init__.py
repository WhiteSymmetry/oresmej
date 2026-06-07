#__init__.py
"""
Oresme, Harmonic Series and Hilbert Space Module (JAX accelerated)
Oresme, Harmonik Seri ve Hilbert Uzayı Modülü (JAX hızlandırmalı)

This module provides:
- Harmonic number calculations (exact fractions and floating point)
- Oresme sequence (n / 2^n) generation
- Hilbert space (ℓ²) membership tests (mathematically sound)
- JAX-accelerated computations for large‑scale work
- Sequence analysis and comparison utilities

Bu modül şunları sağlar:
- Harmonik sayı hesaplamaları (kesirli tam sonuçlar ve kayan noktalı)
- Oresme dizisi (n / 2^n) üretimi
- ℓ² (Hilbert uzayı) aidiyet testleri (matematiksel olarak doğru)
- Büyük ölçekli işlemler için JAX ile hızlandırılmış hesaplamalar
- Dizi analizi ve karşılaştırma yardımcıları
"""

from importlib import import_module
import os
import sys
from typing import List, Union, Optional
import warnings


# JAX kontrolü
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None  # JAX desteklenmiyor

__version__ = "0.1.3"
__author__ = "Mehmet Keçeci <mkececi@yaani.com>"
__license__ = "MIT"

# Dışa aktarılacak semboller listesi
__all__ = [
    'oresme_sequence',
    'harmonic_numbers',
    'harmonic_number',
    'harmonic_number_jax',
    'harmonic_numbers_jax',
    'harmonic_generator_jax',
    'harmonic_number_approx',
    #'harmonic_sum_approx',
    'harmonic_sum_approx_jax',
    'harmonic_convergence_analysis',
    'EULER_MASCHERONI',
    'ApproximationMethod',
    'is_in_hilbert'
]

# Tip tanımları (JAX durumuna göre)
if JAX_AVAILABLE:
    HarmonicSequence = Union[List[float], jnp.ndarray]
else:
    HarmonicSequence = List[float]

# Geliştirme modu ayarı
_DEV_MODE = os.getenv("ORESMEJ_DEV_MODE", "").lower() == "true"

if _DEV_MODE:
    warnings.warn("ORESMEJ: Geliştirme modu aktif", RuntimeWarning)
    import_module('importlib').invalidate_caches()

# Fonksiyonları doğrudan içe aktar
try:
    from .oresmej import (
        oresme_sequence,
        harmonic_numbers,
        harmonic_number,
        harmonic_number_approx,
        EULER_MASCHERONI,
        ApproximationMethod,
        is_in_hilbert
    )

    if JAX_AVAILABLE:
        from .oresmej import (
            harmonic_number_jax,
            harmonic_numbers_jax,
            harmonic_generator_jax,
            #harmonic_sum_approx,
            harmonic_sum_approx_jax,
            harmonic_convergence_analysis
        )
except ImportError as e:
    raise ImportError(
        f"oresmej: Gerekli fonksiyonlar yüklenemedi - {str(e)}"
    ) from e

# Kullanım uyarıları
if sys.version_info < (3, 8):
    warnings.warn(
        "oresmej: Python 3.8+ önerilir. 3.7 desteği v1.0'da kaldırılacak",
        FutureWarning,
        stacklevel=2
    )

if not JAX_AVAILABLE:
    warnings.warn(
        "oresmej: JAX bulunamadı. GPU/TPU hızlandırma devre dışı",
        RuntimeWarning,
        stacklevel=2
    )

# Testler (doğrudan çalıştırılırsa)
if __name__ == "__main__":
    def _test_imports():
        missing = [name for name in __all__ if name not in globals()]
        if missing:
            raise RuntimeError(f"Eksik fonksiyonlar: {missing}")
        print(f"oresmej {__version__} başarıyla yüklendi")
        print(f"JAX desteği: {'AKTİF' if JAX_AVAILABLE else 'PASİF'}")
        print("Örnek çıktı (H(5)):", harmonic_number(5))
    _test_imports()
