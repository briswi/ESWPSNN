# ESWPSNN -- Electron Solar Wind--Plasma Sheet Neural Network
Neural network model that predicts electron flux in the plasma sheet from solar wind inputs.




This repository contains the source code and example usage for the model 
that is described in the paper: \
Swiger et al. (2022). Energetic Electron Flux Predictions in the 
near-Earth Plasma Sheet from Solar Wind Driving. *Space Weather*, (**under 
review**).

This repository is also available on zenodo.org.

[![DOI](https://zenodo.org/badge/489504778.svg)](https://zenodo.org/badge/latestdoi/489504778)


All of the code is written in python; the conda package configuration file is 
`swpsnn.yml`.

The Jupyter Notebook `model_usage_example.ipynb` walks through an example of
how to go from having zero data to having a trained, neural network model.
It shows how to create the model feature arrays (inputs) and model target arrays 
(outputs) from OMNI, FISM-2, and THEMIS data. Then it uses the feature and 
target arrays to train a neural network.
Note that the model that is trained in `model_usage_example.ipynb` is 
only an example.

The full, trained model that is described and analyzed in the 
Swiger et al., 2022 paper is located at 
`Model/swpsnn_v1.2.2.h5`. To open and use it, follow the same steps that 
are shown in Section 4 of `model_usage_example.ipynb`.
The model expects the input array to be in the same format as that created in
Section 2.4 of `model_usage_example.ipynb`. 



