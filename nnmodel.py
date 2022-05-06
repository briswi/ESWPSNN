# --------------------------------------------------------------------------- #
# Written By: Brian Swiger
# Purpose: create class definition for training a neural network
# --------------------------------------------------------------------------- #

from nnconstants import *


def read_pickle(filename):
    """
    Opens a file that was stored using python's pickle module.

    Args:
    -----
    filename        string; full or relative path + filename of the file
                    to be opened

    Returns:
    --------
    object          the underlying object that was stored using pickle.

    """
    import pickle

    f = open(filename, 'rb')
    o = pickle.load(f)
    f.close()

    return o




class NNModel(object):
    """
    NNModel objects contain methods to train neural networks and use
    already trained networks to make predictions.
    """

    def __init__(self,
        features_train_data,
        targets_train_data,
        features_test_data,
        targets_test_data,
        model_dir
        ):
        """
        Args
        ----
        features_train_data     array pre-processed for training
        targets_train_data      array pre-processed for training
        
        features_test_data      array pre-processed for test/validation
        targets_test_data       array pre-processed for test/validation

        model_dir               directory where model output will be saved.

        """

        # Initializing attributes.
        self.model_fldr = model_dir
        self.x_train = features_train_data
        self.y_train = targets_train_data
        self.x_test = features_test_data
        self.y_test = targets_test_data


    def save_as_pickle(self, data, fname):
        """
        Saves *data* to *fname* using python pickle module.

        Args
        ____
        data        any object

        fname       str; file name including path of where the object will
                    be saved

        Returns
        -------
        None
        """

        import pickle

        f = open(fname, 'wb')
        pickle.dump(data, f)
        f.close()

        return None


    def make_predictions(self,
                         model_name, 
                         predict_fname):
        """
        Makes predictions from a previously trained model

        Args:
        -----
        model_name       str; the model description, e.g. "onelayer"

        predict_fname    str; the filename to where the predictions will
                         be stored

        """
        from tensorflow.keras.models import load_model

        # Load the previously trained model.
        saved_model = load_model(model_fname)

        # Use the saved model to make predictions, given the input.
        y_predict = saved_model.predict(self.x_test)

        # Save the predictions.
        self.save_as_pickle(y_predict, predict_fname)

        return None


    def make_single_prediction(self, model_name):
        """
        Makes a prediction using only a subset of the data. Used for making
        predictions of case studies or for specific events.

        Args:
        -----
        model_name       str; the model description, e.g. "onelayer"

        predict_fname    str; the filename to where the predictions will
                         be stored. default=None; if None, will create
                         filename automatically;

        custom_fns       dict; passed automatically by calling method.

        """
        from keras.models import load_model

        # One must specify custom objects that were used to create the model
        # during the training.
        custom_fns = {'bias' : self.bias,
                      'skill' : self.skill,
                      'extremes' : self.extremes,
                      'association' : self.association}

        model_fname, *_, predict_fname = self.create_fnames(model_name)

        # Load the previously trained model.
        saved_model = load_model(model_fname, custom_objects=custom_fns)

        # Make predictions from the input data
        y_predict = saved_model.predict(self.x_train)

        # Save the predictions.
        self.save_as_pickle(y_predict, predict_fname)

        return None


    def train_model(self,
            model_code,
            num_epochs=20,
            do_shuffle=True,
            do_overwrite=False,
            batch_size=100,
            verbosity=1,
            make_reproducible=False
            ):
        """
        Build, train, and save a two layer neural network with keras
        module; predicting electron flux in plasma sheet from solar wind.

        Parameters:
        -----------
        model_code : str
            The code to use for this version of the model, e.g. v0.4.1.

        num_epochs : int, Default=50
            Number of iterations of training from the entire data set.
            

        do_shuffle : bool, default=True              
            Whether to randomly select batches or to take batches sequentially.
            If True, will randomly select each batch from the training
            set. If False, each batch will be selected sequentially.

        do_overwrite : bool, default=False
            Whether to prompt for overwriting an existing saved file.
            If True, then no prompt is given and any previously existing
            save will be overwritten. If False, user is prompted before
            writing final model to disk.

        verbosity : int, default=1
            Verbose options = {0:'silent', 1:'progress_bar', 2:'epoch_line'}

        make_reproducible : bool, default=False
            If set to True, the random seed will be statically set to be able
            to reproduce results between subsequent trainings. Useful for
            development to compare subsequent training runs.

        Returns:
        --------
        None
            Saves the model, model weights, and results to disk, in a location
            created based on *model_code*.
        """

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras import optimizers
        from tensorflow.keras import losses
        from tensorflow.keras import metrics

        if make_reproducible:
            self.set_random_seeds()

        # Create custom filenames.
        save_fname, weights_fname, results_fname = \
            self.create_fnames(model_code)

        # Get some numbers that will help build the layers.
        num_examples = self.x_train.shape[0]
        x_size = self.x_train.shape[1]
        y_size = self.y_train.shape[1]

        # Define model parameters.
        layer1_num_nodes = 832
        hidden_layer_num_nodes = 320
        layer2_num_nodes = y_size
        layer1_activation_fn = 'relu'
        layer2_activation_fn = 'linear'

        # Define optimizer function.
        optimize_fn = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.80142,
            beta_2=0.404999,
            epsilon=1e-7,
            amsgrad=True
            )

        # Define loss function.
        loss_fn = losses.Huber(delta=2.62424)

        # Build the model.
        model = Sequential()
        model.add(
            Dense(
                units=layer1_num_nodes,
                input_shape=(x_size,),
                activation=layer1_activation_fn))

        model.add(
            Dropout(0.41474))

        # Add additional layers.
        # Hidden layer.
        model.add(
            Dense(
                units=hidden_layer_num_nodes,
                activation=layer1_activation_fn))

        model.add(
            Dropout(0.097897))

        # Output layer.
        model.add(
            Dense(
                units=layer2_num_nodes,
                activation=layer2_activation_fn))


        # Configure the learning process.
        model.compile(
            optimizer=optimize_fn,
            loss=loss_fn
            )

        # Train the model.
        results = model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=verbosity,
            validation_data=(self.x_test, self.y_test),
            shuffle=do_shuffle)

        # Save the results for later analysis.
        self.save_as_pickle(results.history, results_fname)

        model.save(save_fname, overwrite=do_overwrite)
        model.save_weights(weights_fname)

        return None


    def set_random_seeds(self, pyseed=123, npseed=456, tfseed=789,
                         hashseed='0'):
        """
        ***There is a problem with this method when using TensorFlow 2.0 ***


        This will fix the random seeds for:

        PYTHONHASHSEED=0
        built-in python random module; random.seed()
        numpy.random.seed()
        tensorflow.compat.v1.set_random_seed()

        It also sets some tensorflow configurations
        so that the stochastic nature of training is
        done the same way each time.
        Notably, it disables multi-threading.

        For further details, see:

        https://keras.io/getting-started/faq/
                #how-can-i-obtain-reproducible-results
                -using-keras-during-development

        https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        https://stackoverflow.com/questions/42022950/

        https://github.com/keras-team/keras/issues/
                2280#issuecomment-306959926

        https://docs.python.org/3.7/using/
                cmdline.html#envvar-PYTHONHASHSEED
        """

        #TODO: remove the print statements and return statement after fixing
        #      the Tensorflow 2.0 multi-threading problem.
        print("When using Tensorflow 2.0, setting session is not available")
        print("Cannot set Tensorflow to use single thread:")
        print("Expect random results.")

        # The following are for setting random seed for reproducibility.
        from os import environ
        from numpy import random as nprand
        import tensorflow.compat.v1 as tfcv1
        import random as rn
        #TODO: determine whether we need to import keras.backend
        #from keras import backend as K

        # Will need to set the PYTHONHASHSEED=0 in order to disable hash
        # randomization. This is similar to invoking it via command line.
        environ['PYTHONHASHSEED'] = hashseed

        # The below is necessary for starting core Python generated
        # random numbers in a well-defined state.
        rn.seed(pyseed)

        # The below is necessary for starting Numpy generated
        # random numbers in a well-defined initial state.
        nprand.seed(npseed)

        # The below tfcv1.set_random_seed() will make random number
        # generation in the TensorFlow backend have a
        # well-defined initial state.
        # For further details, see:
        # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
        tfcv1.set_random_seed(tfseed)

        # Force TensorFlow to use single thread.
        # Multiple threads are a potential source of
        # non-reproducible results.
        # For further details, see:
        # https://stackoverflow.com/questions/42022950/
        session_conf = tfcv1.ConfigProto(intra_op_parallelism_threads=1,
                                         inter_op_parallelism_threads=1)

        sess = tfcv1.Session(graph=tfcv1.get_default_graph(),
                             config=session_conf)


        #TODO: Fix this method to work with Tensorflow 2.0: set_session()
        #      not available when using Tensorflow 2.0
        #K.set_session(sess)


    def create_fnames(self, descr):
        """
        Makes custom folder and file names for the model that is being
        trained, and returns the filenames as strings.

        Will create directories if they do not already exist.

        Args
        ----
        descr       str; the model description, e.g. "onelayer"

        Returns
        -------
        save_fn, weights_fn, results_fn, prediction_fn, observed_fn
        """
        from pathlib import Path

        save_fldr = self.model_fldr + descr + '/SavedModel/'
        results_fldr = self.model_fldr + descr + '/Stats/'
        predictions_fldr = self.model_fldr + descr + '/Predictions/'

        # Create the directories if they do not already exist.
        Path(save_fldr).mkdir(parents=True, exist_ok=True)
        Path(results_fldr).mkdir(parents=True, exist_ok=True)
        Path(predictions_fldr).mkdir(parents=True, exist_ok=True)

        save_fname = save_fldr + descr + '_model.h5'
        weights_fname = save_fldr + descr + '_weights.h5'
        results_fname = results_fldr + descr + '_results.pkl'

        return save_fname,\
               weights_fname,\
               results_fname,\


def optimize_model(
        x_train,
        y_train,
        x_test,
        y_test,
        make_reproducible=True
        ):
    """
    Build, train, and optimize the model using hyperopt.
    hyperopt uses Bayesian optimization to optimize hyperparameters.

    Parameters:
    -----------
    x_train : numpy array
        
    y_train : numpy array

    x_test : numpy array

    y_test : numpy array

    make_reproducible : bool, default=True
        whether to set the random seeds

    Returns:
    --------
    results : dict
        A dictionary of loss, status, and model.

    """

    from numpy import amin
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import optimizers, losses, metrics
    #from tensorflow.keras import callbacks as kcb

    from hyperas.distributions import choice, uniform, quniform, loguniform

    from metrics import MSA_with_log10

    #if make_reproducible:
    #    self.set_random_seeds()


    # Get some numbers that will help build the layers.
    num_examples = x_train.shape[0]
    x_size = x_train.shape[1]
    y_size = y_train.shape[1]

    # Define model parameters.
    outputlayer_num_nodes = y_size

    # Build the model.
    model = Sequential()

    # First hidden layer.
    model.add(
        Dense(
            units={{quniform(32, 1024, 32)}},
            input_shape=(x_size,),
            activation='relu'
            )
            )

    model.add(
        Dropout({{uniform(0, 1)}})),

    # Second hidden layer.
    model.add(
        Dense(
            units={{quniform(32, 1024, 32)}},
            activation='relu',
            )
            )

    model.add(
        Dropout({{uniform(0, 1)}})),

    # Third hidden layer.
    model.add(
        Dense(
            units={{quniform(32, 1024, 32)}},
            activation='relu',
            )
            )

    model.add(
        Dropout({{uniform(0, 1)}})),

    # Output layer.
    model.add(
        Dense(
            units=outputlayer_num_nodes,
            activation='linear'
            )
            )

    # Configure the learning process.
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate={{choice(
                [0.00001, 0.0001, 0.001, 0.01, 0.1])}},
            amsgrad=True,
            beta_1={{uniform(0.001, 0.999)}},
            beta_2={{uniform(0.001, 0.999)}},
            epsilon=1e-7),
        loss=losses.Huber(
            delta={{uniform(0.1, 10.0)}}
            )
        )

    # Train the model.
    results = model.fit(
        x_train,
        y_train,
        batch_size={{choice([100, 500, 1000, 5000, 10000])}},
        epochs=20,
        verbose=0,
        validation_data=(x_test, y_test),
        shuffle=True
        )

    y_modeled = model.predict(x_test)
    # Metric on which to optimize: Median Symmetric Accuracy;
    # see Morley et al., 2018.
    val_msa = MSA_with_log10(y_test, y_modeled)


    return {
        'loss': val_msa, 
        'status': STATUS_OK,
        'model': model
        }
            

