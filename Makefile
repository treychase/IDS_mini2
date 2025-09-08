.PHONY: help install run lint clean

help:
    @echo "Available targets:"
    @echo "  install   Install Python dependencies"
    @echo "  run       Run the driveline.py script"
    @echo "  lint      Lint Python code with flake8"
    @echo "  clean     Remove Python cache files"

install:
    pip3 install --user -r requirements.txt || pip3 install --user pandas matplotlib scikit-learn

run:
    python3 driveline.py

lint:
    flake8 driveline.py || echo "flake8 not installed. Run 'pip3 install flake8' to enable linting."

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
