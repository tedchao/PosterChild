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
    
Install scikit-image for visualization:
    
    pip install scikit-image
    
To convert svg into pdf, simply install cairosvg and run:
    
    pip3 install cairosvg
    cairosvg <image.svg> -o <image.pdf>

Example run:

    python posterization.py images/Kobe.png posterized_images/Kobe.png vectorized_images/Kobe additive_mixing_layers/Kobe0.png
