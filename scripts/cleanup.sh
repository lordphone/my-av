# Run this in the root of the repository

# Cleanup __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Cleanup .pytest_cache directories
find . -type d -name ".pytest_cache" -exec rm -rf {} +

