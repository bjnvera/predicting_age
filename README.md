# Intro
Repo contains code for the project **Predicting age via MRI features** based on the course Advanced Machine Learning at ETH Zurich. 
The goal is to predict the age of a person based on the MRI features. The data is very high-dimensional with 832 features and only 1212 samples.
The final model included a robust scaling of the features, kNN-imputation of missing values, model-based variable selection and gradient boosting as a regression model.
The best way to dig into the repo is to go through the notebooks. 

# Repo structure
Directories are used as follows
- `notebooks` contains jupyter notebooks
  - clear your output before committing
- `data` contains the data used 
  - not included in VCS of git
- `src` contains the source code, possibly shared among multiple files and notebooks. 
