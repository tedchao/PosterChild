# posterization

## Installation

### macOS or Linux

Use [Poetry](https://python-poetry.org/). Install Poetry itself via `pip install poetry` or `brew install poetry`. Then:

    poetry install --no-root
    poetry shell

To create a double-clickable .app for macOS users, run (inside a poetry shell):

    pyinstaller posterization_gui.spec

A double-clickable `Posterization GUI.app` will appear inside the `dist/` directory.

### Windows

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to the posterization directory.
Then:

    conda env create -f environment.yml
    conda activate posterization

## Usage

Launch the GUI:

    python src/gui.py

Command line example:

    python src/cli.py images/obama.jpg posterized_images/obama additive_mixing_layers/obama
