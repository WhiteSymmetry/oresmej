#__init__.py
"""
oresmej package initialization

This module serves as the main entry point for the oresmej package,
handling version control, imports, and compatibility warnings.
"""

import os
import sys
import warnings
from typing import List, Union, Optional
from importlib import import_module

# JAX kontrolü
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None  # JAX desteklenmiyor

__version__ = "0.1.0"
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
    'harmonic_sum_approx',
    'harmonic_sum_approx_jax',
    'harmonic_convergence_analysis',
    'EULER_MASCHERONI',
    'ApproximationMethod',
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
        ApproximationMethod
    )

    if JAX_AVAILABLE:
        from .oresmej import (
            harmonic_number_jax,
            harmonic_numbers_jax,
            harmonic_generator_jax,
            harmonic_sum_approx,
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


"""
# Sorunsuz

import os
import sys
import warnings
from typing import List, Union, Optional
from importlib import import_module

# JAX ve NumPy importları
try:
    import jax.numpy as jnp
    import numpy as np
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    import numpy as np
    jnp = None  # JAX desteklenmiyor

__version__ = "0.1.0"
__author__ = "----"
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
    'harmonic_sum_approx',
    'harmonic_sum_approx_jax',
    'harmonic_convergence_analysis',
    'EULER_MASCHERONI',
    'ApproximationMethod',
]

# Tip tanımları (JAX durumuna göre)
if JAX_AVAILABLE:
    HarmonicSequence = Union[List[float], jnp.ndarray, np.ndarray]
else:
    HarmonicSequence = Union[List[float], np.ndarray]

# Geliştirme modu ayarı
_DEV_MODE = os.getenv("ORESMEJ_DEV_MODE", "").lower() == "true"

if _DEV_MODE:
    warnings.warn("ORESMEJ: Geliştirme modu aktif", RuntimeWarning)
    importlib.invalidate_caches()


def _import_from(module_name: str, names):
    # Dinamik modül yükleme fonksiyonu.
    module = import_module(module_name, package=__name__)
    return [getattr(module, name) for name in names]


try:
    # Temel fonksiyonları yükle
    base_names = [
        'oresme_sequence',
        'harmonic_numbers',
        'harmonic_number',
        'harmonic_number_approx',
        'EULER_MASCHERONI',
        'ApproximationMethod'
    ]

    # Göreceli import: oresmej.py'den temel fonksiyonlar
    from .oresmej import (
        oresme_sequence,
        harmonic_numbers,
        harmonic_number,
        harmonic_number_approx,
        EULER_MASCHERONI,
        ApproximationMethod
    )

    # globals() güncelle
    base_funcs = [eval(name) for name in base_names]
    globals().update(zip(base_names, base_funcs))

    # JAX fonksiyonlarını yükle (mevcutsa)
    if JAX_AVAILABLE:
        from .oresmej import (
            harmonic_number_jax,
            harmonic_numbers_jax,
            harmonic_generator_jax,
            harmonic_sum_approx,
            harmonic_sum_approx_jax,
            harmonic_convergence_analysis
        )

        jax_names = [
            'harmonic_number_jax',
            'harmonic_numbers_jax',
            'harmonic_generator_jax',
            'harmonic_sum_approx',
            'harmonic_sum_approx_jax',
            'harmonic_convergence_analysis'
        ]

        jax_funcs = [eval(name) for name in jax_names]
        globals().update(zip(jax_names, jax_funcs))

except Exception as e:
    raise ImportError(
        f"oresmej fonksiyonları yüklenemedi. "
        f"Paketin doğru kurulduğundan emin olun.\n"
        f"Hata detayı: {str(e)}"
    ) from e


# Kullanım uyarıları
if sys.version_info < (3, 8):
    warnings.warn(
        "ORESMEJ: Python 3.8+ önerilir. 3.7 desteği v1.0'da kaldırılacak",
        FutureWarning,
        stacklevel=2
    )

if not JAX_AVAILABLE:
    warnings.warn(
        "ORESMEJ: JAX bulunamadı. GPU/TPU hızlandırma devre dışı",
        RuntimeWarning,
        stacklevel=2
    )


# Testler (doğrudan çalıştırılırsa)
if __name__ == "__main__":
    def _test_imports():
        # İçe aktarılan fonksiyonları doğrular
        missing = [name for name in __all__ if name not in globals()]
        if missing:
            raise RuntimeError(f"Eksik fonksiyonlar: {missing}")
        
        print(f"oresmej {__version__} başarıyla yüklendi")
        print(f"JAX desteği: {'AKTİF' if JAX_AVAILABLE else 'PASİF'}")
        print("Örnek çıktı (H(5)):", harmonic_number(5))
    
    _test_imports()
"""    

"""
import os
import sys
import importlib
import warnings
from typing import List, Union, Optional
import jax.numpy as jnp

# JAX kontrolü
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

__version__ = "0.1.0"
__author__ = "Mehmet Keçeci <mkececi@yaani.com>"
__license__ = "MIT"

__all__ = [
    'oresme_sequence',
    'harmonic_numbers',
    'harmonic_number',
    'harmonic_number_jax',
    'harmonic_numbers_jax',
    'harmonic_generator_jax',
    'harmonic_number_approx',
    'harmonic_sum_approx',
    'harmonic_sum_approx_jax',
    'harmonic_convergence_analysis',
    'EULER_MASCHERONI',
    'ApproximationMethod',
    #'HarmonicSequence'
]

# Geliştirme modu ayarı
_DEV_MODE = os.getenv("ORESMEJ_DEV_MODE", "").lower() == "true"

if _DEV_MODE:
    warnings.warn("ORESMEJ: Geliştirme modu aktif", RuntimeWarning)
    importlib.invalidate_caches()

# Ana fonksiyonları içe aktar
try:
    from .oresmej import (
        oresme_sequence,
        harmonic_numbers,
        harmonic_number,
        harmonic_number_approx,
        EULER_MASCHERONI,
        ApproximationMethod
    )
    
    # JAX fonksiyonları koşullu içe aktarım
    if JAX_AVAILABLE:
        from .oresmej import (
            harmonic_number_jax,
            harmonic_numbers_jax,
            harmonic_generator_jax,
            harmonic_sum_approx,
            harmonic_sum_approx_jax,
            harmonic_convergence_analysis
        )
        
except ImportError as e:
    raise ImportError(
        f"oresmej: Temel fonksiyonlar yüklenemedi - {str(e)}"
    ) from None

# Tip tanımları
if JAX_AVAILABLE:
    HarmonicSequence = Union[List[float], jnp.ndarray]
else:
    HarmonicSequence = List[float]

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
        # İçe aktarılan fonksiyonları doğrular
        missing = [name for name in __all__ if not globals().get(name)]
        if missing:
            raise RuntimeError(f"Eksik fonksiyonlar: {missing}")
        
        print(f"oresmej {__version__} başarıyla yüklendi")
        print(f"JAX desteği: {'AKTİF' if JAX_AVAILABLE else 'PASİF'}")
    
    _test_imports()

"""

"""
# sorunsuz
from __future__ import annotations  # Gelecekteki özellikler için (Python 3.7+)

import importlib
import warnings
import os
if os.getenv("DEVELOPMENT") == "true":
    importlib.reload(oresmej)


# Göreli modül içe aktarmaları
# F401 hatasını önlemek için sadece kullanacağınız şeyleri dışa aktarın
# Aksi halde linter'lar "imported but unused" uyarısı verir
try:
    from .oresmej import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    from . import oresmej  # Modülün kendisine doğrudan erişim isteniyorsa
except ImportError as e:
    warnings.warn(f"Gerekli modül yüklenemedi: {e}", ImportWarning)


# Eski bir fonksiyonun yer tutucusu - gelecekte kaldırılacak
def eski_fonksiyon():

    # Kaldırılması planlanan eski bir fonksiyondur.
    # Lütfen alternatif fonksiyonları kullanın.

    warnings.warn(
        "eski_fonksiyon() artık kullanılmamaktadır ve gelecekte kaldırılacaktır. "
        "Lütfen yeni alternatif fonksiyonları kullanın. "
        "Oresme Python 3.9-3.14 sürümlerinde desteklenmektedir.",
        category=DeprecationWarning,
        stacklevel=2
    )


# Paket sürüm numarası
__version__ = "0.1.0"


# Geliştirme sırasında test etmek için
if __name__ == "__main__":
    eski_fonksiyon()
"""
