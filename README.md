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

Example run:

    python posterization.py images/Kobe.png images/post_Kobe.png [output_vectorized_image_name]
