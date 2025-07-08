

# Install newest versions of the libraries
conda create -n 2025_PeekNuclei python=3.10 pandas numpy tifffile scikit-image scipy matplotlib seaborn

# Export versions that were installed to yaml file
# conda env export -n 2025_PeekNuclei > 2025_PeekNuclei_env.yaml

# Install from yaml file
# conda env create -f 2025_PeekNuclei_env.yaml