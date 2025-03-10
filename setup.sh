#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install Jupyter kernel for this virtual environment
pip install ipykernel
python -m ipykernel install --user --name=well_ts_env --display-name="Well Time Series Environment"

echo "Setup complete! You can now start Jupyter notebook with 'jupyter notebook'" 