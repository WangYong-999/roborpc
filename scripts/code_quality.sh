#!/bin/bash

# Navigate to the directory containing the 'roborpc' folder
# cd /path/to/your/code/directory

# Run autopep8 with max aggressiveness (-aaa) and in-place modification (-i)
# on all Python files (*.py) under the 'roborpc' directory.
autopep8 --in-place --aggressive --aggressive --recursive --experimental --list-fixes package/

# Run black with default settings, since black does not have an aggressiveness level.
# Black will format all Python files it finds in the 'roborpc' directory.
black --experimental-string-processing package/

# Run ruff on the 'roborpc' directory.
# Add any additional flags if needed according to your version of ruff.
ruff --unsafe_fix

# YAPF
yapf --recursive --in-place --verbose --style=google --parallel package
