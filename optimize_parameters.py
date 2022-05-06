# -------------------------------------------------------------------------- #
# Written By: Brian Swiger
# Purpose: train nn using NNModel class
# -------------------------------------------------------------------------- #

import pickle
from pandas import read_hdf

#from numpy import amin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers, losses, metrics

from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, quniform, loguniform
from hyperas import optim

from nnmodel import optimize_model
from metrics import MSA_with_log10

# Change this.
version = "example"


def get_data():
    """
    Load data for NN training.
    """

    # Define folders/directories.

    # Need to change the model version here manually.
    data_dir = './Data/Training/ModelReady/' 
    data_dir = './ExampleData/' 

    # Define data filenames.
    ftrain_fname = data_dir + 'train_features.h5'
    ttrain_fname = data_dir + 'train_targets.h5'
    ftest_fname = data_dir + 'val_features.h5'
    ttest_fname = data_dir + 'val_targets.h5'

    # Load the training data.
    x_train = read_hdf(ftrain_fname).values
    y_train = read_hdf(ttrain_fname).values
    x_test = read_hdf(ftest_fname).values
    y_test = read_hdf(ttest_fname).values

    return x_train, y_train, x_test, y_test, True


def optimize(
    max_iters=10,
    print_results=True,
    save_dir='./Data/Optimizations/'
    ):
    """
    Optimizes hyperparameters.

    Parameters:
    -----------
    max_iters : int, default 10
        the number of iterations to try while optimizing
        
    print_results : bool, default True
        whether to print results immediately

    save_dir : str,
        default is './Data/Optimizations/'

    Returns:
    --------
        None; saves the results to disk at save_dir

    """
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    best_run, best_model = optim.minimize(
        model=optimize_model,
        data=get_data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
        eval_space=True  #Saves the 'best' parameters by object repr
        )

    # Uncomment to see results immediately.
    _, _, X_test, Y_test, _ = get_data()
    if print_results:
        print("Evaluation of best performing model:")
        print(best_model.evaluate(X_test, Y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)

    # Saving the results to disk.
    run_file = open(save_dir + "best_run_" + version + ".pkl", "wb")
    pickle.dump(best_run, run_file)
    run_file.close()

    best_model.save(save_dir + "best_model_" + version + ".h5")

    


