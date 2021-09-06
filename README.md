# PosterChild

PosterChild: Blend‐Aware Artistic Posterization (EGSR 2021). Please read [paper](https://cragl.cs.gmu.edu/posterchild/PosterChild-%20Blend-Aware%20Artistic%20Posterization%20(Cheng-Kang%20Ted%20Chao,%20Karan%20Singh,%20Yotam%20Gingold%202021%20EGSR)%20300dpi.pdf) for more details.

[Cheng-Kang Ted Chao, Karan Singh, Yotam Gingold.]

## License

This work is dual-licensed under Apache 2.0 and MIT.

`SPDX-License-Identifier: Apache-2.0 and MIT`

## Installation

### MacOS or Linux

Use [Poetry](https://python-poetry.org/). Install Poetry itself via `pip install poetry` or `brew install poetry`. Then:

    poetry install --no-root
    poetry shell

To create a double-clickable .app for macOS users, run (inside a poetry shell):

    pyinstaller posterization_gui.spec

A double-clickable `Posterization GUI.app` will appear inside the `dist/` directory.

### Windows or MacOS

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to the posterization directory.
Then:

    conda env create -f environment.yml
    conda activate posterization

To update an already created environment if the `environment.yml` file changes, first activate and then run `conda env update --file environment.yml --prune`.

## Usage

Launch the GUI:

    python src/gui.py

Command line example:

    python src/cli.py [source-image-path] [output-image-path] [output-additive-mixing-path] [0 or 1 (indicating if using downsampled approach)]
