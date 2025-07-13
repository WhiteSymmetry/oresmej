# Oresme Jax

[![DOI](https://zenodo.org/badge/DOI/-.svg)](https://doi.org/-)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI--blue)](https://doi.org/)

[![figshare DOI](https://img.shields.io/badge/DOI-1-blue)](https://doi.org/)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmej/badges/version.svg)](https://anaconda.org/bilgi/oresmej)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmej/badges/latest_release_date.svg)](https://anaconda.org/bilgi/oresmej)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmej/badges/platforms.svg)](https://anaconda.org/bilgi/oresmej)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/oresmej/badges/license.svg)](https://anaconda.org/bilgi/oresmej)
[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Oresme numbers refer to the sums related to the harmonic series.

---
### **Türkçe Tanım:**
**Oresme Sayıları**, 14. yüzyılda Nicole Oresme tarafından incelenen matematiksel serilerdir. Oresme sayıları harmonik seriye ait toplamları ifade eder. İki türü vardır:  
1. **\( \frac{n}{2^n} \) serisi** (Oresme'nin orijinal çalışması),  
2. **Harmonik sayılar** (\( H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \)).  
Bu sayılar, analiz ve sayı teorisinde önemli rol oynar.

---

### **English Definition:**
**Oresme Numbers** are mathematical series studied by Nicole Oresme in the 14th century. Oresme numbers refer to the sums related to the harmonic series. They include two types:  
1. The **\( \frac{n}{2^n} \) sequence** (Oresme's original work),  
2. **Harmonic numbers** (\( H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \)).  
These numbers play a key role in analysis and number theory.

---

### **Fark/Karşılaştırma (Difference):**
- **Oresme'nin \( \frac{n}{2^n} \) serisi** ıraksaklık kanıtları için önemlidir.  
- **Harmonik sayılar** (\( H_n \)) ise logaritmik büyüme gösterir ve \( n \to \infty \) iken ıraksar.  
- Modern literatürde "Oresme numbers" terimi daha çok tarihsel bağlamda kullanılır.

---

## Kurulum (Türkçe) / Installation (English)

### Python ile Kurulum / Install with pip, conda, mamba
```bash
pip install oresmej -U
python -m pip install -U oresmej
conda install bilgi::oresmej -y
mamba install bilgi::oresmej -y
```

```diff
- pip uninstall Oresme -y
+ pip install -U oresmej
+ python -m pip install -U oresmej
```

[PyPI](https://pypi.org/project/Oresme/)

### Test Kurulumu / Test Installation

```bash
pip install -i https://test.pypi.org/simple/ oresmej -U
```

### Github Master Kurulumu / GitHub Master Installation

**Terminal:**

```bash
pip install git+https://github.com/WhiteSymmetry/oresmej.git
```

**Jupyter Lab, Notebook, Visual Studio Code:**

```python
!pip install git+https://github.com/WhiteSymmetry/oresmej.git
# or
%pip install git+https://github.com/WhiteSymmetry/oresmej.git
```

---

## Kullanım (Türkçe) / Usage (English)

```python
import oresmej as oj 



```

```python
import oresmej
oresmej.__version__
```
---

### Development
```bash
# Clone the repository
git clone https://github.com/WhiteSymmetry/oresmej.git
cd oresmej

# Install in development mode
python -m pip install -ve . # Install package in development mode

# Run tests
pytest

Notebook, Jupyterlab, Colab, Visual Studio Code
!python -m pip install git+https://github.com/WhiteSymmetry/oresmej.git
```
---

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX


### APA

```
Keçeci, M. (2025). Dynamic vs Static Number Sequences: The Case of Keçeci and Oresme Numbers. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15833351

Keçeci, M. (2025). oresmej (0.1.0). Open Science Articles (OSAs), Zenodo. https://doi.org/
```

### Chicago

```
Keçeci, Mehmet. Dynamic vs Static Number Sequences: The Case of Keçeci and Oresme Numbers. Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.15833351

Keçeci, Mehmet. Oresme. Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/

```


### Lisans (Türkçe) / License (English)

```
This project is licensed under the MIT License.
```
