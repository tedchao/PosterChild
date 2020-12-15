# posterization

## Installation

### macOS or Linux

Use [Poetry](https://python-poetry.org/). Install Poetry itself via `pip install poetry` or `brew install poetry`. Then:

    poetry install --no-root
    poetry shell

To create a double-clickable .app for macOS users, run (inside a poetry shell):

    pyinstaller posterization_gui.spec

### Windows

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version.
Then:

    conda env create -f environment.yml
    conda activate posterization
    pip install git+https://github.com/yig/pyGCO@buil-win

## Usage

Example run:

    python posterization.py images/obama.jpg posterized_images/obama additive_mixing_layers/obama
    
Our run it to play with GUI:

    python posterization_gui.py
