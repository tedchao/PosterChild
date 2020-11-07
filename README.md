# posterization

## Installation

Use [Poetry](https://python-poetry.org/). Install Poetry itself via `pip install poetry` or `brew install poetry`. Then:

    poetry install --no-root
    poetry shell

## Usage

Install potrace:

    brew install libagg pkg-config potrace
    git clone https://github.com/flupke/pypotrace.git
    cd pypotrace
    pip install .

Install cairo:
    
    pip install pycairo
    
Install scikit-image for visualizing palette:
    
    pip install scikit-image
    
To convert svg into pdf, simply install cairosvg and run:
    
    pip3 install cairosvg
    cairosvg <image.svg> -o <image.pdf>

Example run:

    python posterization.py images/obama.jpg posterized_images/obama additive_mixing_layers/obama
