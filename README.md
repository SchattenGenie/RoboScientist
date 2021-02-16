# RoboScientist

```info
author: tba
```

## Installation and usage

This repository is set up as a proper python package, thus it can be installed by:
```sh
git clone git@github.com:SchattenGenie/RoboScientist.git
cd roboscientist

# installs the package
pip install -e .
```

Take into account `-e` flag in the last command --- it tells pip to install the package in "editable" mode:
all changes to the project files immediately take effect without need to reinstall the package (this, however, does not affect already imported code in a running interpreter).
Nevertheless, adding/removing top-level modules (ones directly under `src/`) still requires reinstallation.

## Run the code

In `examples`-directory there are snippets notebooks with minimal working examples for formula learning.