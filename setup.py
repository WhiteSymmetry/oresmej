from setuptools import setup, find_packages

setup(
    name="oresmej",
    use_scm_version=True,  # Sürüm bilgisini setuptools_scm ile alır
    setup_requires=["setuptools", "wheel", "setuptools_scm"],  # Gerekli kurulum bağımlılıkları
    version='0.1.0',
    packages=find_packages(where="src"),  # src dizinindeki modülleri bul
    package_dir={"": "src"},  # src dizinine yönlendirme
    include_package_data=True,  # Ek dosyaları dahil et
    install_requires=[
    	"numpy",
        ],
    author="Mehmet Keçeci",
    description="Oresme numbers refer to the sums related to the harmonic series.",
    url="https://github.com/oresmej",
    license="MIT",
    python_requires='>=3.9',
)
