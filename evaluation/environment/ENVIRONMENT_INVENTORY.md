# Environment Inventory

Audit date: 2026-09-18  
Operating system observed by Python: Windows 10 build 22631

## StarDist environment

- path: `<CONDA_ENVS>\cellproject\python.exe`
- Python 3.10.20
- csbdeep 0.8.2
- matplotlib 3.10.9
- NumPy 2.2.6
- pandas 2.3.3
- Pillow 12.3.0
- scikit-image 0.25.2
- scikit-learn 1.7.2
- SciPy 1.15.3
- StarDist 0.9.2
- TensorFlow 2.21.0

## Cellpose environment

- path: `<CONDA_ENVS>\cellpose_env\python.exe`
- Python 3.10.21
- Cellpose 3.1.1.3
- NumPy 2.0.2
- pandas 2.3.3
- Pillow 12.3.0
- scikit-image 0.25.2
- scikit-learn 1.7.2
- SciPy 1.15.3

## Default system Python

- path: `<SYSTEM_PYTHON>\python.exe`
- Python 3.11.9
- core scientific packages were not installed at audit time

## LaTeX

- bundled Tectonic: 0.17.0, minimal smoke test passed
- unavailable on PATH: TeX Live, `latexmk`, `pdflatex`, `xelatex`, `lualatex`, `biber`, `kpsewhich`

The exact package lists will be exported to lock/evidence files before the corrected evaluator runs. Different pipeline stages must record the interpreter path used.
