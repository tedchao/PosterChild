# posterization

## Installation

Use [Poetry](https://python-poetry.org/). Install Poetry itself via `pip install poetry` or `brew install poetry`. Then:

    poetry install --no-root
    poetry shell

## Usage

Example run:

    python posterization.py images/obama.jpg posterized_images/obama additive_mixing_layers/obama
    
Our run it to play with GUI:

    python posterization_gui.py
