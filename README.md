### Intro
Repo used for developing code for AML project predicting age via MRI features

### Setup
It is recommended to
- work in a virtual environment (venv)
   
1. Update pip
```
python -m pip install --upgrade pip
``` 

2. Install dependencies given in 
```
python -m pip install -r requirements.txt
``` 

### Repo structure
Directories are used as follows
- `notebooks` contains jupyter notebooks
  - clear your output before committing
- `data` contains the data used 
  - not included in VCS of git
- `src` contains the source code, possibly shared among multiple files and notebooks. 
- On top level are scripts such as `src/print_hello_world`. One level below, in packages, files containing functionalities used among multiple scripts and notebooks, e.g. `/src/dummy/hello_world.py`.

To run a script, go to `src/` and execute it the file with python, e.g. `python ./print_hello_world.py` 

### Test the setup

1. Run the script located in `./src/print_hello_world.py`
2. Run successfully the jupyter notebook `./notebooks/hello_world.ipynb`
 
