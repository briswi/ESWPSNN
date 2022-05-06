# --------------------------------------------------------------------------- #
# Written By: Brian Swiger
# Purpose: define classes and methods for data that will be used to train
#          a neural network
# --------------------------------------------------------------------------- #

from nnconstants import *


def calc_mlt(x, y):
    """Given x and y coordinates, return the MLT value."""

    from numpy import arctan2, pi

    # Using the negative of (x, y) because of the way the GSM coordinate
    # system is related to MLT, where midnight is at (-x, y=0) and
    # dawn (MLT 06) is at (x=0, -y).
    radians = arctan2(-y, -x)

    # The returned domain of arctan2 is [-pi, pi); we want [0, 2pi).
    radians[radians<0] += 2*pi

    return radians * 12 / pi  # There are 24 hours of MLT, and 2pi*12/pi = 24.


def calc_xy_dist(x, y):
    """Given x and y coordinates, calculate a distance"""

    return (x**2 + y**2)**(0.5)






def save_data(data, fname):
        """
        Saves *data* to *fname* as a python pickle object in binary format.
        """
        import pickle

        try:
            f = open(fname, 'wb')

        except:
            f = open('./misplacedfile.dat', 'wb')

        pickle.dump(data, f)
        f.close()

        return None


def get_dtstring(dtime, *args):
    """
    returns string representations for a given datetime or pd.Timestamp
    object;

    Arguments
    ---------
    dtime        datetime.datetime or equivalent (e.g., pandas.Timestamp)

    *args        any combination of:

                "ymd", "year", "month", "day", "hour", "minute"

                 as a sequence of strings


    Returns
    -------
    time_dict    a tuple with strings that are the string
                 representation of the passed *args
                 (zero padded for all except year)

    Example
    -------
    >>> dtime = datetime.datetime(2020, 10, 3, 9, 23)
    >>> get_dtstring(dtime, "ymd", "month", "hour")
    ('20201003', '10', '09')

    """

    fullstr = dtime.strftime('%Y%m%d%H%M')
    time_dict = {
                'ymd' : fullstr[:8],
                'year' : fullstr[:4],
                'month' : fullstr[4:6],
                'day' : fullstr[6:8],
                'hour' : fullstr[8:10],
                'minute' : fullstr[10:]
                }
    time_strings = [time_dict[arg] for arg in args]

    return tuple(time_strings)


def display_remaining_time(current_step, total_steps, step_size, begin_time):
    """
    Prints the estimated remaining time before a loop finishes.

    Parameters
    ----------
    current_step : int
        the current step in the loop

    total_steps : int
        the total number of steps in the loop

    step_size : int
        the step size of the iterator in the loop

    begin_time : datetime.datetime object
        the time that the loop was started, usually provided by a call to
        datetime.datetime.now()

    Returns
    -------
    None    prints the estimated time to screen and returns None.

    ** Note that begin_time should be calculated outside the loop. **
    """
    import datetime

    remain_steps = total_steps - current_step
    elapsed_time = datetime.datetime.now() - begin_time
    remain_time = elapsed_time * remain_steps / (current_step * step_size)

    remain_minutes = remain_time.seconds // 60
    if remain_minutes != 0:
        remain_seconds = \
            remain_time.seconds % (remain_minutes*60)
    else:
        remain_seconds = remain_time.seconds

    print("Estimated time remaining: {} min, {} sec".format(
        remain_minutes, remain_seconds))

    return None


class TrainingData(object):
    """
    Describes the training data as input data and output data that is used
    to train a neural network.
    """

    def __init__(self, day1=None, day2=None, fname=None,
                 single_event=False):
        """
        Instantiates the object by opening and loading into memory the
        file that is found at fname.
        """

        self.begin_date = str(day1)
        self.end_date = str(day2)
        self.single_event = single_event

        if fname is not None:
            self.data = self._load_data(fname)

        else:
            self.data = None


    def __str__(self):
        """
        The string representation of the object.
        """
        #TODO Implement str method.

    def __repr__():
        """

        """
        #TODO Implement repr method.



    def get_data(self, copy=False):
        """
        Return the data contained within this object, it will be a
        pandas DataFrame.

        Args:
        -----
        copy        bool; whether to explicitly make a copy the DataFrame;
                    useful if the returned DataFrame will be modified;
                    default=False

        Returns:
        --------
        DataFrame   the data associated with the object
        """

        return self.data.copy(copy)


    def _load_data(self, fname):
        """
        Opens and loads the file at fname.
        (Typically, this is a serialized pickle object previously saved as
        a pandas DataFrame.)

        Returns None if the file does not exist.
        """

        import pandas as pd

        try:
            return pd.read_pickle(fname)

        except FileNotFoundError:
            return None


    def download_cdf(self, base_path='', mission='', level='',
                     datatype='', date='', spacecraft=''):
        """
        Returns a string that will be the url where the data are saved
        on the ftp server.

        *** default for all inputs is empty string; ''

        Inputs:
        -------
        base_path:      str; base path to where cdf file should be stored

        mission:        str; name of mission, see
                        ftp://cdaweb.gsfc.nasa.gov/pub/data/

        level:          str; 'l0', 'l1', 'l2', etc.

        datatype:       str; instrument or other

        date:           str; in 'yyyymmdd' format

        spacecraft:     spacecraft name if mult-spacecraft mission
                        e.g. use tha, thb, etc. for THEMIS A, THEMIS B,
                        rbspa, rbspb for Van Allen Probes A, B
                        default: empty string

        returns:
        --------
        None            downloads the cdf file to the location specified in
                        base_path, returns None
        """

        from pathlib import Path
        import urllib.request as req

        # Make the file_path where the file is on the server & where it will be
        # saved locally.
        file_path = ''
        for fp in [mission, spacecraft, level, datatype, date[:4]]:
            if fp != '':
                file_path += fp + '/'


        # Make the filename that is on the server; and will be used locally.
        file_name = ''

        if mission == 'omni':
            if datatype == 'hourly':
                datatype = 'h0_mrg1hr'
                mission = 'omni2'
            for fn in [mission, datatype, date]:
                if fn != '':
                    file_name += fn + '_'
            file_name += 'v01.cdf'

        elif mission == 'themis' and level == 'ssc':
            file_name = spacecraft + '_or_ssc_' + date + '_v01.cdf'

        else:
            for fn in [spacecraft, level, datatype, date]:
                if fn != '':
                    file_name += fn + '_'
            file_name += 'v01.cdf'


        # Write the new directory locally if needed.
        Path(base_path + file_path).mkdir(parents=True, exist_ok=True)

        # Check to see if the file is already downloaded.
        if Path(base_path + file_path + file_name).exists():
            return None


        # Add on each part and return url; this is without the filename.
        full_url  = BASE_URL + file_path

        err_smt = 'Error in fetching url: {}'.format(full_url + file_name)

        # Get response from server; or print error message if file not found.
        try:
            response = req.urlopen(full_url + file_name)
        except Exception as e:
            #print(err_smt)
            #print(e)
            return None

        # Read the response and save it as a file.
        diskfile = open(base_path + file_path + file_name, 'wb')
        diskfile.write(response.read())
        diskfile.close()

        response.close()



    def calc_vector_mag(self, *args):
        """
        given a vector field in n number of orthogonal directions,
        1, 2, 3, 4, etc., calculate and return the magnitude of the
        n-dimensional vector

        Parameters
        ----------
        args :  1D array-like or scalar; n number of vector field arrays
            in n directions

        Returns
        -------
        vector_mag : same type, shape as args, the magnitude of the
            vector using all provided components


        Examples
        --------
        # Using arrays.
        x1, x2 = np.random.random_sample((5,)), np.random.random_sample((5,))
        x_mag = calc_vector_mag(x1, x2)
        # x_mag will be an array of shape (5,)

        -------------------------------------------------------------------
        # Using scalars.
        x, y, z = 30.9, 12.4, -2.3
        r = calc_vec_mag(x, y, z)
        """
        magnitude_squared = 0.0
        for arg in args:
            magnitude_squared += arg**2

        return magnitude_squared**0.5


    def filter_features(self,
        features_filename,
        themis_positions_filename,
        save_filename,
        num_hours_history=4,
        avg_minutes=5,
        lag_minutes=10
        ):
        """
        Uses previously found Timestamps of THEMIS mission plasma sheet
        observations and combined input features to create a new DataFrame
        containing the time history of each feature for each THEMIS Timestamp.
        The new DataFrame is sliced and saved to disk in yearly segments.
        The yearly segmenting is to keep the file sizes relativily low.
        Saving is performed via pickle.

        Also adds columns of sin(phi), cos(phi), rdist, based on the THEMIS
        position data.

        phi = MLT * (2pi / 24)
        rdist = (Rdist - Rdist_mu) / Rdist_sigma
        Rdist = sqrt(X^2 + Y^2) (in GSM coordinates)
        Rdist_mu = Rdist sample mean
        Rdist_sigma = Rdist sample standard deviation

        rdist is the normalized Rdist.

        Parameters
        ----------
        features_filename : str
            filename, including directory, where the features DataFrame is
            stored; the data in this DataFrame are expected to have
            been already normalized, and OMNI features interpolated

        themis_positions_filename : str
            filename, including directory, where THEMIS plasma sheet filtered
            positions are stored; found by running the
            'filter_data' method of ThemisData

        save_filename : string
            specify the path to save the created DataFrames. 
            filename with extension '.pkl' as if no modifications will be
            made. Example: save_filename="save_dir/save_filename.pkl"

            The method will then save by year with pattern:
            "save_dir/save_filename_yyyy.pkl"

        num_hours_history : numeric (int or float)
            number of hours of parameter history data to use
            in relation to each THEMIS observation; default=4

        avg_minutes : numeric (int or float)
            number of minutes that the parameter data are to be averaged;
            default=5

        lag_minutes : numeric (int or float)
            lag time, in minutes, of parameter data assumed to influence
            plasma sheet variations. At typical magnetic field and plasma
            densities, and assuming influence travels at the Alfven speed,
            it takes variations ~20-30 minutes to go ~20-30Re
            (bow shock to plasma sheet). Various studies / authors have found
            that a delay of ~10 minutes is appropriate, i.e. Wang et al., 2017;
            default=10


        Returns:
        --------
        None            Saves combined DataFrame in yearly slices to disk
                        at location specified by save_filename with the year
                        appended to each filename.
        """
        import datetime
        from numpy import empty, linspace, pi, cos, sin
        from pandas import DataFrame, read_pickle
        from pathlib import Path

        # Checking to see whether the file already exists.
        if Path(save_filename).exists():
            print("File {} already exists.".format(save_filename))
            choice = ''
            while choice == '':
                choice = input("Do you want to overwrite y/n? ")
            if 'n' in choice.casefold():
                return None

        # Open and read feautures and THEMIS position data.
        features = read_pickle(features_filename)
        themis_positions = read_pickle(themis_positions_filename)

        # Set up the resulting DataFrame
        time_lag = datetime.timedelta(minutes=lag_minutes)
        num_inputs = features.shape[1] \
                   * ((num_hours_history * 60 - lag_minutes) \
                   // avg_minutes + 1)

        # Time object representing number of hours before as timedelta object.
        hours_previous = datetime.timedelta(hours=num_hours_history)

        # Create column names for the storage DataFrame.
        column_names = list()
        num_repeats = num_inputs // features.shape[1]
        for tdelay in linspace(num_hours_history*60, lag_minutes, num_repeats):
            for p in features.columns:
                column_names.append(p + '-{}'.format(int(tdelay)))

        # Create a new empty DataFrame with defined column names. Each row
        # will be labeled with the corresponding datetime from the THEMIS
        # positions DataFrame.
        num_samples = themis_positions.shape[0]
        master_inputs = DataFrame(
            empty([num_samples, num_inputs]),
            columns=column_names,
            index=themis_positions.index
            )

        # Go through each THEMIS Timestamp; estimate the time remaining in loop.
        counter = 0
        query_step_size = num_samples // 20
        loop_begin_time = datetime.datetime.now()
        for dtime in themis_positions.index:

            # Initial estimated time remaining.
            if counter == 100:
                display_remaining_time(
                    counter,
                    num_samples,
                    1,
                    loop_begin_time
                    )

            # Periodic estimated time remaining.
            if counter % query_step_size == 0 and counter > 0:
                display_remaining_time(
                    counter,
                    num_samples,
                    1,
                    loop_begin_time)

            # Begin creating this timestamps features.
            dtime_prvs = dtime - hours_previous

            # Slice the feature df to include only hours_previous hours
            # up to the dtime minus a time lag.
            # (Using assumption that info at t0 does not affect result at t0.)
            features_window = features.loc[dtime_prvs : dtime-time_lag, :]

            # Average the slice to avg_minutes minutes, ignoring NaNs.
            features_window_avg = features_window.resample(
                            str(avg_minutes)+'T').mean()

            # Convert the dataframe into a flattened numpy array object.
            # Flattened array will be part of input training example.
            features_flat = features_window_avg.values.flatten()

            # Add training input to master input DataFrame.
            master_inputs.loc[dtime, :] = features_flat

            counter += 1


        # Add sin(phi), cos(phi), rdist to master_inputs.
        phi = themis_positions.mlt * pi / 12
        master_inputs['cos_phi'] = cos(phi)
        master_inputs['sin_phi'] = sin(phi)
        master_inputs['rdist'] = \
            (themis_positions.rdist - themis_positions.rdist.mean()) \
              / themis_positions.rdist.std()

        # We need to periodically save because the complete DataFrame
        # is too large to be pickled. We choose periods of one year.
        print("Master Inputs DataFrame is using {} GiB of memory".format(
            master_inputs.memory_usage(index=True, deep=True).sum()/1e9))

        first_year = master_inputs.index[0].year
        last_year = master_inputs.index[-1].year
        ext = save_filename[-4:]
        ext_idx = save_filename.rfind(ext)

        for year in range(first_year, last_year+1, 1):
            # Create a filename unique for this year.
            save_filename_thisyear = \
                save_filename[:ext_idx] \
                + '_' \
                + str(year) \
                + ext

            # Slice the master inputs for this year.
            master_inputs_thisyear = master_inputs.loc[str(year), :]

            # Save the master input array as a pickled object.
            save_data(master_inputs_thisyear, save_filename_thisyear)
            print("Saved data for year {}.".format(year))


        return None


    def create_training_arrays_by_year(self,
        features_dir,
        targets_dir,
        save_dir,
        training_splits
        ):
        """
        Organizes the final data structures for training the NN.

        Using the yearly features (inputs) and targets (outputs) data files,
        removes any entries from both that have missing data.

        Then, use *training_splits* to concatenate the yearly features and
        targets arrays into single arrays for training, testing, and validation.



        Parameters
        ----------
        features_dir : str
            the directory where the features data are stored.

        targets_dir : str
            the directory where the targets data are stored.

        save_dir : str
            the directory where the resulting arrays will be saved.

        training_splits : dict
            dictionary whose keys are ['train', 'test', 'val'], and values are
            lists of years. Example:
            training_splits={'train':[2007, 2008, 2010],
                             'test':[2009],
                             'val':[2011]}

        Returns:
        --------
        None
            saves the resulting arrays in *save_dir* using filename format:
            ("<type>_features.h5", "<type>_targets.h5")
            where <type> is one of 'train', 'test', or 'val'.

        """

        from pandas import read_pickle, concat

        for key in training_splits.keys():

            # Create save filenames.
            feature_filename = save_dir + key + '_features.h5'
            target_filename = save_dir + key + '_targets.h5'

            features = []
            targets = []
            for year in training_splits[key]:

                yearly_feature_fname = \
                    features_dir \
                    + "feature_data_with_themis_filter_noesasat_" \
                    + str(year) \
                    + ".pkl"

                yearly_target_fname = \
                    targets_dir \
                    + "themis_plasmasheet_noesasat_nflux_" \
                    + str(year) \
                    + ".pkl"

                features.append(read_pickle(yearly_feature_fname))
                targets.append(read_pickle(yearly_target_fname))

            features = concat(features)
            targets = concat(targets)

            # Drop any feature row that contains NaN from both features and
            # targets. (The targets already have all NaNs removed.)
            feature_rows_with_nan = features.isna().sum(axis=1) > 0
            features = features.loc[~feature_rows_with_nan, :]
            targets = targets.loc[~feature_rows_with_nan, :]

            assert len(features.index) == len(targets.index), \
                "Features and Targets for {} have different indices.".format(
                key)

            features.to_hdf(feature_filename, key=key+'_features')
            targets.to_hdf(target_filename, key=key+'_targets')

        return None


    def create_training_arrays_by_fraction(self,
        features_fname,
        targets_fname,
        save_dir,
        training_splits
        ):
        """
        Organizes the final data structures for training the NN.

        Using the features (inputs) and targets (outputs) data files,
        removes any entries from both that have missing data.

        Then, use *training_splits* to concatenate the yearly features and
        targets arrays into single arrays for training, testing, and validation.



        Parameters
        ----------
        features_fname : str
            where the features data are stored.

        targets_fname : str
            where the targets data are stored.

        save_dir : str
            the directory where the resulting arrays will be saved.
            default filenames will be created 

        training_splits : dict
            dictionary whose keys are ['train', 'test', 'val'], and values are
            fractions that add to 1. Example:
            training_splits={'train': 0.8,
                             'test': 0.1,
                             'val': 0.1}

        Returns:
        --------
        None
            saves the resulting arrays in *save_dir* using filename format:
            ("<type>_features.h5", "<type>_targets.h5")
            where <type> is one of 'train', 'test', or 'val'.

        """

        from pathlib import Path
        from math import floor
        from pandas import read_pickle, concat
        from numpy import random

        # Create the directory if necessary.
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        
        # Load the feature and target data.
        feature_data = read_pickle(features_fname)
        target_data = read_pickle(targets_fname)

        num_examples = target_data.shape[0]

        # Create a randomly generated permutated array of ints of length
        # the number of rows in the targets df.
        rng = random.default_rng()
        permutation = rng.permutation(num_examples)

        # Take the requested fraction of items from the permutation 
        # sequence for each split.
        num_train = floor(num_examples * training_splits['train'])
        num_val = floor(num_examples * training_splits['val'])
        num_test = floor(num_examples * training_splits['test'])

        split_sequences = {
            'train' : permutation[:num_train],
            'val' : permutation[num_train:num_train+num_val],
            'test' : permutation[-num_test:]
            }

        for key in training_splits.keys():

            # Create save filenames.
            feature_filename = save_dir + key + '_features.h5'
            target_filename = save_dir + key + '_targets.h5'

            features = feature_data.take(
                split_sequences[key]
                )
            
            targets = target_data.take(
                split_sequences[key]
                )

            # Drop any feature row that contains NaN from both features and
            # targets. (The targets already have all NaNs removed.)
            feature_rows_with_nan = features.isna().sum(axis=1) > 0
            features = features.loc[~feature_rows_with_nan, :]
            targets = targets.loc[~feature_rows_with_nan, :]

            assert len(features.index) == len(targets.index), \
                "Features and Targets for {} have different indices.".format(
                key)

            features.to_hdf(feature_filename, key=key+'_features')
            targets.to_hdf(target_filename, key=key+'_targets')

        return None




# --------------------------------------------------------------------------- #
class FismData(TrainingData):
    """some description"""

    def __init__(self):
        """description..."""

    def download_fism_data(self,
        time1,
        time2,
        energy_range,
        save_dir=FISM_DATA_FLDR,
        data_format='csv',
        ):

        """download FISM2 data from server at LaTiS, part of LASP.

        Parameters
        ----------
        time1 : datetime.datetime or similar
        time2 : datetime.datetime or similar

        save_dir : str

        data_format : str

        energy_ranges : str

        """

        from pathlib import Path
        import urllib.request as req

        #TODO include support for downloading individual bands using the
        # energy_ranges arg.

        time_format = '%Y-%m-%dT%H:%M:%S.000Z'
        time1_str = time1.strftime(time_format)
        time2_str = time2.strftime(time_format)

        filename = save_dir + 'fism2_' + time1_str + '_' + time2_str \
                 + '_' + energy_range + '.' + data_format

        base_url = "https://lasp.colorado.edu/lisird/latis/dap/"
        format_url = base_url + "fism_flare_bands." + data_format

        full_url = format_url + "?&time>=" + time1_str + "&time<=" \
            + time2_str + '&time,' + energy_range


        # Write the new directory locally if needed.
        Path(save_dir).mkdir(parents=True, exist_ok=True)


        # Check to see if the file is already downloaded.
        if Path(filename).exists():
            return None

        err_smt = 'Error in fetching url: {}'.format(full_url)

        # Get response from server; or print error message if file not found.
        try:
            response = req.urlopen(full_url)
        except Exception as e:
            #print(err_smt)
            #print(e)
            return None

        # Read the response and save it as a file.
        diskfile = open(filename, 'wb')
        diskfile.write(response.read())
        diskfile.close()

        response.close()

    def dparser(self, s):
        """

        """
        from pandas import Timestamp

        return Timestamp.utcfromtimestamp(int(s))

    def extract_fism_data(self,
        time1,
        time2,
        csv_dir = FISM_DATA_FLDR,
        save_dir = FISM_DATA_FLDR,
        ):
        """


        """
        from pandas import read_csv
        from numpy import float64

        time_format = '%Y-%m-%dT%H:%M:%S.000Z'
        time1_str = time1.strftime(time_format)
        time2_str = time2.strftime(time_format)

        fname_format = "fism2_" + time1_str + "_" + time2_str + "_E54_0_65_0"
        csv_fname = csv_dir + fname_format + ".csv"

        fismdata = read_csv(
            csv_fname,
            header=0,
            dtype=float64,
            names=['twenty_eV'],
            index_col=0,
            parse_dates=True,
            date_parser=self.dparser,
            )

        # Save as pickle.
        save_fname = save_dir + fname_format + '.pkl'
        fismdata.to_pickle(save_fname)

        return None




    def normalize_fism(self, data):
        """

        """

        #TODO: finish method.

    def add_fism_to_features(self, fism_data, features_array):
        """

        """

        #TODO: finish method.


class ThemisData(TrainingData):
    """
    Contains methods and attributes for downloading, extracting, storing,
    and preparing THEMIS mission data for neural network training
    """

    def __init__(self, day1=None, day2=None, mlt_bin=None, radial_bin=None,
                 fname=None, single_event=False):
        """

        """
        self.begin_date = day1
        self.end_date = day2
        self.mltbin = str(mlt_bin)
        self.radbin = str(radial_bin)
        self.single_event = single_event

        if fname is not None:
            self.data = super()._load_data(fname)

        elif day1 is not None:
            self.data = self._load_data()

        else:
            self.data = None


    def _load_data(self):
        """
        Returns the THEMIS data as a pandas DataFrame that has been previously
        saved for dates from *day1* to *day2*.


        Returns:
        --------
        themis_data       DataFrame; the data that is stored in the file
                          Returns None if the file does not exist.
        """
        from pathlib import Path

        if self.single_event:
            fldr = THEMIS_DATA_FLDR + 'SingleEvents/'
            fname = fldr + 'themis_mlt' + self.mltbin + '_' \
                  + self.begin_date + '.pkl'

        else:
            fldr = THEMIS_DATA_FLDR + 'Mlt' + self.mltbin + '/Rad' \
                 + self.radbin + '/'

            fname = fldr + 'themis_mlt' + self.mltbin + '_rad' + self.radbin \
                  + '_' + self.begin_date + '_' + self.end_date + '.pkl'

        if Path(fname).exists():
            return super()._load_data(fname)

        else:
            return None


    def get_begdate(self):
        """
        Returns the first date of THEMIS data as a string in format "yyyymmdd"
        """
        return self.begin_date


    def get_enddate(self):
        """
        Returns the last date of THEMIS data as a string in format "yyyymmdd"
        """
        return self.end_date




    def __str__(self):
        """

        """
        #TODO Implement str for ThemisData class.

    def __repr__(self):
        """

        """
        #TODO Implement repr for ThemisData class.

    def download_cdf(self,
        dates,
        datatype,
        probe,
        dnload_fldr=CDF_BASE_FLDR,
        ):
        """
        Downloads the requested CDF files from the NASA GSFC SPDF server
        if the file does not already exist either on the local machine,
        or on the CIFS server. If on the CIFS server, copies from CIFS;
        if on neither, downloads the CDF from NASA.

        Args
        ----
        dates:          two-element tuple or list or single string;
                        string representations of
                        the dates that are requested for the files in
                        YYYYMMDD format. (e.g. ['20100101', '20100201'] will
                        download one month worth of data, '20100101' will
                        download a single day.)

        datatype:       string; for THEMIS data this can be either 'gmom',
                        'fgm', etc. or 'ssc' depending on which instrument
                        dataset requesting

        probe:          string or list of strings; the probe designation from
                        the THEMIS mission. e.g. 'tha' or 'thd'
                        or ['tha', 'thd', 'the'].

        dnload_fldr:    string; the start of the path to save the CDF files;
                        Defaults to CDF_BASE_FLDR; recommend to not change.
                        The CDF will be downloaded into a file organization
                        system that mirrors the ftp server at NASA GSFC SPDF.


        Returns
        -------
        None            prints error message to stdout for any CDF files that
                        were not able to download. Otherwise, saves CDF file
                        to disk in default location.

        """

        from pandas import date_range

        mission = 'themis'

        #'D' frequency means Day and is one absolute day
        #'MS' frequency means month start
        #see:
        #http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        implemented_datasets = ['gmom', 'fit', 'fgm']

        if datatype == 'ssc':
            level = datatype
            datatype = ''
            date_freq = 'MS'

        elif datatype in implemented_datasets:

            level = 'l2'
            date_freq = 'D'

        else:
            msg = 'datatype {} not yet implemented'.format(datatype)
            print(msg)
            return None

        # Check whether a single date was passed or requesting multiple dates.
        if len(dates) == 2:
            date_array = date_range(dates[0], dates[1], freq=date_freq)

        elif level == 'ssc':
            date = dates[:-2] + '01'
            date_array = date_range(date, date, freq=date_freq)

        else:
            date_array = date_range(dates, dates, freq=date_freq)

        if type(probe) == str:
            probe = [probe]

        for p in probe:

            # This is used for printing message to screen.
            current_year, = get_dtstring(date_array[0], 'year')

            for d in date_array:
                ymd, year = get_dtstring(d, 'ymd', 'year')

                # Print out when we go into a new year.
                if year != current_year:
                    msg = 'Finished downloading THEMIS-{} for {}.'.format(
                          p[-1].capitalize(), current_year)

                    print(msg)
                    current_year = year

                # Download the CDF.
                super().download_cdf(dnload_fldr, mission, level,
                                     datatype, ymd, p,)


    def extract_data(self,
        dates,
        probes,
        save_fldr=THEMIS_DATA_FLDR,
        cdf_fldr=CDF_BASE_FLDR,
        ):
        """
        Extracts data from CDF files and builds Pandas DataFrame objects.
        Saves the DataFrame to disk as a pickled python object.

        If the CDF file does not already exist on the local machine, then
        download_cdf() is called in an attempt to download.

        Args
        ----
        dates:          two-element tuple or list or single string;
                        string representations of
                        the dates that are requested for the files in
                        YYYYMMDD format. (e.g. ['20100101', '20100201'] will
                        extract one month worth of data, '20100101' will
                        extract only the single day.)


        probes:         string or list of strings; the probe designation from
                        the THEMIS mission. e.g. 'tha' or 'thd'
                        or ['tha', 'thd', 'the'].

        save_fldr:      string; the start of the path to save the pickle files;

        cdf_fldr:       string; the start of the path to look for the CDF files;
                        Defaults to CDF_BASE_FLDR; recommend to not change.
                        The CDF files are in a file organization that
                        mirrors the ftp server at NASA GSFC SPDF.


        Returns
        -------
        None            prints error message to stdout for any CDF files that
                        were not able to open. Otherwise, saves pickle file
                        to disk in save_fldr location.

        """
        from pathlib import Path
        import pandas as pd

        # Extract cdfs between these dates.
        # 'D' frequency means Day and is one absolute day.
        # See:
        # pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        if len(dates) == 2:
            date_array = pd.date_range(dates[0], dates[1], freq='D')

        else:
            date_array = [pd.Timestamp(dates)]

        if type(probes) == str:
            probes = [probes]


        for date in date_array:

            ymdstr, year, month = get_dtstring(date, 'ymd', 'year', 'month')

            for probe in probes:
                # Append to the base folder for this date and spacecraft.
                themis_fldr = save_fldr + probe + '/' + year + '/'

                # Write the file path if it does not already exist.
                Path(themis_fldr).mkdir(parents=True, exist_ok=True)

                # Name the output file.
                themis_fname = themis_fldr + probe + '_' + ymdstr \
                             + '_combined.pkl'

                # Check to verify the file does not exist before continuing.
                if Path(themis_fname).exists():
                    continue


                # Set the CDF filename and make sure it exists. If it does
                # not exist, download it.
                gmom_cdf = cdf_fldr + 'themis/' + probe + '/l2/gmom/' \
                         + year + '/' + probe + '_l2_gmom_' \
                         + ymdstr + '_v01.cdf'

                ssc_cdf = cdf_fldr + 'themis/' +  probe + '/ssc/' \
                        + year + '/' + probe + '_or_ssc_' \
                        + year + month + '01_v01.cdf'


                if not Path(gmom_cdf).exists():
                    self.download_cdf(
                        ymdstr,
                        'gmom',
                        probe,
                        )

                if not Path(ssc_cdf).exists():
                    self.download_cdf(
                        ymdstr,
                        'ssc',
                        probe,
                        )

                # Make the DataFrame; _build_themis_df() returns False if
                # there was an error encountered.
                df = self._build_themis_df(
                    gmom_cdf,
                    ssc_cdf,
                    probe,
                    )

                if df is not False:
                    # After extraction, save dataframe as pickled object.
                    save_data(df, themis_fname)


    def _build_themis_df(self, gmom_fname, ssc_fname, probe):
        """
        opens THEMIS cdf files and extracts specific variables to make a pandas
        DataFrame with a DatetimeIndex and column names given by either cdf
        variable names or derived from them in the case of multi-dimensional
        data (e.g. pterf_magf has three columns, x, y, z)


        Inputs:
        -------
        gmom_fname      str; the name, including full path, of the gmom
                        (ground calculated moments) file


        ssc_fname       str; the name, including full path, of the ssc
                        (spacecraft position) file

        probe:          str; the three letter designation for which spacecraft
                        e.g., 'tha'


        Returns:
        --------
        combined_df     a dataframe with energy flux columns, density, velocity,
                        spacefraft position, and many other columns;
                        the index is a datetime index by gmom data;
                        the s/c position has a lower cadence than gmom, and is
                        matched as closely as possible to the more frequent
                        gmom cadence
                        the fit data has similar resolution as gmom data but
                        does not always match the time-stamps exactly, used
                        a 'nearest' algorithm to sync the gmom and fit data
        """
        import pandas as pd
        from spacepy import pycdf

        # Open gmom cdf, if it exists.
        try:
            gmom = pycdf.CDF(gmom_fname)

        except Exception as e:
            #print('Could not open', gmom_fname)
            #print(e)
            return False


        # Start with getting the data from the gmom file.
        # The data returned by spacepy.pycdf.CDF() is in a numpy array.
        fluxvals = gmom[probe + '_pterf_en_eflux'][...]
        flux_channel_array = gmom[probe + '_pterf_en_eflux_yaxis'][...]

        # Encountered a .cdf file from themis that had empty eflux data.
        try:
            flux_channels = flux_channel_array[0,:].round(1)

        except Exception as e:
            print('CDF file: {} is likely missing eflux data'.format(
                                                              gmom_fname))
            print(e)
            return False

        # The gmom time is given as seconds since 1970-01-01 00:00:00.
        time = gmom[probe + '_pterf_time'][...]
        time_index = pd.to_datetime(time, unit='s')

        # Builds the initial dataframe for flux values.
        flux_df = pd.DataFrame(fluxvals,
                               index=time_index,
                               columns=flux_channels.astype('str'))

        # Keep getting data and add it to the DF.
        flux_df['eDensity'] = gmom[probe + '_pterf_density'][...]
        flux_df['eTemp'] = gmom[probe + '_pterf_avgtemp'][...]
        flux_df['edataq'] = gmom[probe + '_pterf_data_quality'][...]
        velocity = gmom[probe + '_pterf_velocity_gsm'][...]
        magf = gmom[probe + '_pterf_magf'][...]
        flux_df['VX_GSM'] = velocity[:, 0]
        flux_df['VY_GSM'] = velocity[:, 1]
        flux_df['VZ_GSM'] = velocity[:, 2]
        flux_df['BX_DSL'] = magf[:, 0]
        flux_df['BY_DSL'] = magf[:, 1]
        flux_df['BZ_DSL'] = magf[:, 2]


        # Discovered at least one CDF that had duplicate entries in time array.
        if flux_df.index.has_duplicates:
            # Keeps only the first occurrence of each time.
            flux_df = flux_df.loc[~flux_df.index.duplicated(), :]

        # Sorting index if needed, discovered that at least one CDF had time
        # array that was not monotonically increasing.
        if not flux_df.index.is_monotonic:
            flux_df.sort_index(inplace=True)

        # The time for the ion array is different than the electron time array.
        i_density = gmom[probe + '_ptirf_density'][...]
        i_temp = gmom[probe + '_ptirf_avgtemp'][...]
        i_dataq = gmom[probe + '_ptirf_data_quality'][...]
        i_time = gmom[probe + '_ptirf_time'][...]

        # Close gmom cdf file.
        gmom.close()

        # Make a DatetimeIndex for the ion data.
        i_time_index = pd.to_datetime(i_time, unit='s')

        # Make the ion DataFrame.
        ion_df = pd.DataFrame(i_density,
                              columns=['ionDensity'],
                              index=i_time_index)

        ion_df['ionTemp'] = i_temp
        ion_df['ionDataQ'] = i_dataq

        # Sorting index if needed, discovered that at least one CDF had time
        # array that was not monotonically increasing.
        if not ion_df.index.is_monotonic:
            ion_df.sort_index(inplace=True)

        # Discovered at least one CDF that had duplicate entries in time array.
        if ion_df.index.has_duplicates:
            # Keeps only the first occurrence of each time.
            ion_df = ion_df.loc[~ion_df.index.duplicated(), :]

        # Match the index of the ion_df to the index of the flux_df,
        # which uses the electron time array as the index.
        ion_df_reindexed = ion_df.reindex(index=flux_df.index,
                                          method='nearest')

        # Now merge the ion data into the flux data frame.
        flux_ion_df = pd.concat([flux_df, ion_df_reindexed], axis=1)


        # ------------------------- #
        # Open ssc (orbit) cdf file.
        try:
            ssc = pycdf.CDF(ssc_fname)

        except Exception as e:
            #print('Could not open', ssc_fname)
            #print(e)
            return False

        # Get the spacecraft location.
        gsm_coords = ssc['XYZ_GSM'][...]

        # The time axis for the orbit position is different than that for gmom.
        gsm_time = ssc['Epoch'][...]

        # Turn the array of datetime objects into a pandas DatetimeIndex object.
        index_time = pd.DatetimeIndex(gsm_time)

        # Make a dataframe for the orbit position data.
        orbit_df = pd.DataFrame(gsm_coords,
                                columns=['X_GSM', 'Y_GSM', 'Z_GSM'],
                                index=index_time)

        # Add columns with more position data to the orbit dataframe.
        orbit_df['gsmLongitude'] = ssc['GSM_LON'][...]
        orbit_df['gsmLatitude'] = ssc['GSM_LAT'][...]

        # Close ssc cdf file.
        ssc.close()
        # ------------------------- #
        # The orbit data is for an entire month; need to use only the day that
        # matches the flux data.

        # Take the date from the filename.
        fluxday = gmom_fname[-16:-8]

        # Only use the day that the flux data are from.
        orbit_df_day = orbit_df.loc[fluxday, :]

        # The orbit cadence is longer than the flux cadence;
        # will reindex orbit_df to match that of the flux_df.
        orbit_df_reindexed = orbit_df_day.reindex(index=flux_df.index,
                                                  method='nearest')

        # Combine the flux and orbit data frames together.
        combined_df = pd.concat([flux_ion_df, orbit_df_reindexed],
                                axis=1, sort=False)


        return combined_df


    def add_fit_data(self, day1, day2, probes, cdf_dir):
        """
        Adds magnetic field in GSM to a themis combined dataframe, that
        has been already created and saved.

        day1            str; in yyyymmdd format
        day2            str; in yyyymmdd format

        probes          list of which probes to use, i.e. ['tha', 'thd', 'the']

        cdf_dir         str; the directory to look for CDF files.
                        This should be the head directory

        day1 and day2 are the start and stop days, inclusive, of the dates to
        apply using this method.

        returns         None

        This method saves the new dataframe to disk.
        """

        from pathlib import Path
        import pandas as pd
        from spacepy import pycdf

        all_dates = pd.date_range(day1, day2, freq='D')

        for probe in probes:

            for date in all_dates:
                ymd, year = get_dtstring(date, 'ymd', 'year')

                # Create the filenames for the FIT CDF, flux df, and new df
                fit_fname = cdf_dir + 'themis/' + probe + '/l2/fit/' \
                          + year + '/' + probe + '_l2_fit_' \
                          + ymd + '_v01.cdf'

                fluxdf_fname = '../../ProjectData/NNData/Themis/' \
                             + probe + '/' + year + '/' + probe \
                             + '_' + ymd + '_combined.pkl'

                flux_fit_fname = '../../ProjectData/NNData/Themis/' \
                             + probe + '/' + year + '/' + probe \
                             + '_' + ymd + '_combined_withfit.pkl'

                # Only continue if the filename does not already exist
                if Path(flux_fit_fname).exists():
                    continue

                try:
                    fit = pycdf.CDF(fit_fname)

                except Exception as e:
                    #print('Could not open', fit_fname)
                    #print(e)
                    continue

                # Get the magnetic field data.
                mag_field = fit[probe + '_fgs_gsm'][...]
                mag_time = fit[probe + '_fgs_time'][...]

                # We are done with the FIT CDF.
                fit.close()

                # Create a dataframe for the FIT data.
                fit_df = pd.DataFrame(mag_field,
                                      index=pd.to_datetime(mag_time, unit='s'),
                                      columns=['BX_GSM', 'BY_GSM', 'BZ_GSM'])

                # Sorting index if needed
                if not fit_df.index.is_monotonic:
                    fit_df.sort_index(inplace=True)

                # Discovered at least one CDF that had duplicate entries.
                if fit_df.index.has_duplicates:
                    # Keeps only the first occurrence of each time.
                    fit_df = fit_df.loc[~fit_df.index.duplicated(), :]

                # Get the flux dataframe
                try:
                    flux_df = pd.read_pickle(fluxdf_fname)
                except Exception as e:
                    #print("Could not open {}".format(fluxdf_fname))
                    #print(e)
                    continue


                # Match the index of the ion_df to the index of the flux_df.
                fit_df_reindexed = fit_df.reindex(index=flux_df.index,
                                                  method='nearest')

                # Combine the FIT and flux dataframes.
                flux_plus_fit_df = pd.concat([flux_df, fit_df_reindexed],
                                             axis=1, sort=False)

                # Save the FIT and flux combined dataframe to disk as h5.
                flux_plus_fit_df.to_pickle(flux_fit_fname)

        return None

    def filter_data(self,
            themis_data,
            beta_threshold=None,
            keep_greater_beta=True,
            mlt_bounds=None,
            rdist_bounds=None,
            drop_extra_cols=True
            ):
        """
        Filters THEMIS data by plasma beta, radial distance, MLT

        Parameters
        ----------
        themis_data : DataFrame,
            Themis data that has been combined using the
            combine_dfs() method.

        beta_threshold : float or None,
            Threshold for which the plasma beta needs to be greater
            than in order to keep the data. Default is None, which is to
            not filter for plasma beta
            The lower and upper thresholds for which to filter data by
            radial distance. Radial distance is defined as r=sqrt(x**2 + y**2)
            Default is None, which is to not filter by radial distance.

        keep_greater_beta : bool, default True,
            If True, keep observations where the plasma beta is greater than
            the *beta_threshold*. Otherwise, keep observations where the plasma
            beta is less than or equal to *beta_threshold*.
            Ignored if *beta_threshold* is None.

        mlt_bounds : tuple-like of shape (2,) or None,
            The lower and upper bounds for magnetic local time (lower, upper).
            Default is None, which is to not filter by MLT.

        rdist_bounds : tuple-like of shape (2,) or None,
            The lower and upper bounds for radial distance (lower, upper).
            Default is None, which is to not filter by Radial Distance.

        drop_extra_cols : bool,
            Whether to drop columns that were used to filter the data,
            keeping only the flux data columns. If False, then the returned
            DataFrame includes all original columns plus those calculated in
            order to filter as specified.
            Default is True, which is to return only flux data columns.

        Returns
        -------
        filtered_df : DataFrame,
            a dataframe with the same number of columns (parameters) as
            the original but with rows (timestamps) removed depending on
            criteria passed.
        """

        data = themis_data.copy()

        # Remove rows with any NaN or inf values.
        from pandas import options
        options.mode.use_inf_as_na = True

        data.dropna(inplace=True)

        # Keep only rows where beta > passed value.
        if beta_threshold is not None:
            if keep_greater_beta:
                data = data.loc[data.beta > beta_threshold, :]
            else:
                data = data.loc[data.beta <= beta_threshold, :]

        # Recalculate MLT values based on X, Y GSM values.
        if mlt_bounds is not None:
            data.loc[:, 'mlt'] = self.calc_mlt(data.X_GSM, data.Y_GSM)

            # Only keep rows where MLT is between (lower, upper) values.
            mlt_bool = (data.mlt >= mlt_bounds[0]) | \
                       (data.mlt <= mlt_bounds[1])

            data = data.loc[mlt_bool, :]

        # Calculate column of radial distance.
        if rdist_bounds is not None:
            data.loc[:, 'rdist'] = \
                super().calc_vector_mag(data.X_GSM, data.Y_GSM)

            # Remove all rows that are not within requested (lower, upper)
            # radial distance bounds.
            rdist_bool = (data.rdist >= rdist_bounds[0]) & \
                         (data.rdist <= rdist_bounds[1])

            data = data.loc[rdist_bool, :]


        # If drop extra columns is True, drop all non-flux data columns.
        # Otherwise, return all columns.
        if drop_extra_cols:
            extra_cols = ['mlt', 'mlt_value', 'beta', 'probe', 'rdist',
                'X_GSM', 'Y_GSM', 'Z_GSM']

            data.drop(extra_cols, inplace=True, axis=1)

        # Return the filtered dataframe.
        return data


    def combine_dfs(self,
            beg_date,
            end_date,
            save_dir,
            data_dir,
            dataq=None,
            probes=['tha', 'thd', 'the']
            ):
        """
        Combines THEMIS data in daily dataframes into single container.

        Args:
        -----
        beg_date        string; beginning date of THEMIS DataFrames to combine;
                        use format 'yyyymmdd'

        end_date        string; ending date of THEMIS DataFrames to combine;
                        use format 'yyyymmdd'

        save_dir       string; specify the path to save the
                       resulting combined DataFrame.

        data_dir       directory where themis daily dataframes are located.

        dataq : 1D array-like, scalar or None,
            sequence of data quality flags that are acceptable to use data.
            The Default is None, which is to not filter by data quality.
            Examples are 0, which uses only data with data quality flags
            marked as "good data". Other options might include, for example,
            [0, 256], where 256 = counter saturation detected in SST data.
            For a list of all data quality flags and their meanings, see:
            http://themis.ssl.berkeley.edu/gmom_flag.shtml

        probes          which probes to use; this should be sequence, even if
                        there is only one, i.e., ['the'] or ('tha',).
                        default is ['tha', 'thd', 'the'].


        Returns:
        --------
        None            Saves combined DataFrame to disk at location specified
                        by save_dir with file name
                        'themis_yyyymmdd_yyyymmdd.pkl'
        """
        from pathlib import Path
        import pandas as pd
        from numpy import log10, nan

        # Specify date range for the files that will be collated; 'D' is daily.
        date_array = pd.date_range(beg_date, end_date, freq='D')

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = save_dir + 'themis_' \
                 + beg_date + '_' + end_date + '.pkl'

        if Path(filename).exists():
            return None

        themis_df_list = []
        # Open each pickled DataFrame and use only rows that match the criteria.
        for date in date_array:

            year, ymd = get_dtstring(date, 'year', 'ymd')

            for probe in probes:
                current_file = data_dir + probe \
                             + '/' + year + '/' + probe \
                             + '_' + ymd + '_combined.pkl'

                try:
                    data = pd.read_pickle(current_file)
                except Exception as E:
                    #print('Could not read', current_file)
                    #print(E)
                    continue

                # Rename some of the energy column labels to make them
                # standard across all spacecraft.
                # Encountered more than one THEMIS CDF file that had abnormal
                # energy channel labels; they were different from the others,
                # and they contained NaNs. In the DataFrames, we previously
                # converted np.nan to string type ('nan')
                if 'nan' in data.columns:
                    # Do not use this DataFrame; it is likely currupted.
                    # But print to screen, so we know which ones.
                    print("Did not use (nan in column labels):")
                    print(current_file)
                    continue

                # Setting the Themis-E probe to be the standard for labeling.
                if probe != 'the':
                    column_swaps = dict(zip(ENERGY_CHNLS[probe],
                                            ENERGY_CHNLS['the']))
                    data.rename(inplace=True, columns=column_swaps)


                # Account for the possibility of having missing data.
                for k in THEMIS_FILL_VALUES.keys():
                    data.replace(THEMIS_FILL_VALUES[k], nan, inplace=True)

                if dataq is not None:
                    # Only keep rows that have acceptable data quality flags.
                    # Data quality flags are found at:
                    # themis.ssl.berkeley.edu/gmom_flag.shtml
                    all_good_data_bools = self._determine_good_quality(data,
                                                                       dataq)

                    data = data.loc[all_good_data_bools, :]

                # Get the plasma beta and mlt values for this time series,
                # and add them as columns to the dataframe.
                plasma_beta = self._calc_plasma_beta(data)
                mlt_values = self._calc_mlt_value(data)

                data['beta'] = plasma_beta
                data['mlt_value'] = mlt_values

                # Round the DatetimeIndex to seconds (they are in microseconds,
                # and there is no need to be that accurate).
                data.index = data.index.round('S')

                # Round the data to 1 minute cadence instead of the 3 second
                # cadence in order to save memory. In resampling, we will
                # take the mean of the observations.
                resampled_df = data.resample('T').mean()


                # Take the log (base 10) of the flux columns.
                try:
                    logflux = \
                    resampled_df.loc[:, ENERGY_CHNLS['use'] ].apply(log10)

                except Exception as e:
                    print(e)
                    print("Did not use (log10 fail):\n{}".format(current_file))
                    continue

                target_df = pd.concat([logflux,
                                       resampled_df.beta,
                                       resampled_df.mlt_value,
                                       resampled_df.X_GSM,
                                       resampled_df.Y_GSM,
                                       resampled_df.Z_GSM],
                                       axis=1, sort=False)


                # Add a column for which probe these data are from.
                target_df['probe'] = probe

                # Add the final result to the list of filtered DataFrames.
                themis_df_list.append(target_df)


        # Combine all of the Data Frames together row-wise.
        combined_df = pd.concat(themis_df_list, axis=0, sort=False)
        combined_df.sort_index(inplace=True)

        # Save the final dataframe to filename.
        save_data(combined_df, filename)

        return None



    def _determine_good_quality(self,
            themis_df,
            good_flags
            ):
        """
        returns a boolean array where the acceptable quality flags are given by
        True and unacceptable by False, for each row of the THEMIS DataFrame

        Args
        ----
        themis_df       pandas DataFrame of themis data extracted from CDF file

        good_flags      sequence of ints representing acceptable flags;
                        see themis.ssl.berkeley.edu/gmom_flag.shtml for
                        flag descriptions


        Returns
        -------
        good_data_bools  pandas boolean Series with index matching the index
                         of input the data frame

        """
        # The data quality flags for the themis gmom data are included in the
        # original DataFrame.
        e_qflags = themis_df.edataq
        i_qflags = themis_df.ionDataQ

        # Make boolean arrays for each of the acceptable flags, then combine
        # using logical OR. There are separate flags for ions and electrons.
        good_ebools = [(e_qflags == q) for q in good_flags]
        good_ibools = [(i_qflags == q) for q in good_flags]

        combo_ebool = good_ebools[0]
        combo_ibool = good_ibools[0]
        for i in range(1, len(good_ebools)):
            combo_ebool = combo_ebool | good_ebools[i]
            combo_ibool = combo_ibool | good_ibools[i]


        good_data_bools = combo_ebool & combo_ibool

        return good_data_bools.values



    def calc_xy_dist(self, x, y):
        """
        Given *x* and *y* coordinates, return a radial distance.

        """

        return (x**2 + y**2)**(0.5)


    def _calc_plasma_beta(self, themis_df):
        """
        calculates and returns a Series of plasma beta for time series of
        a data frame of THEMIS spacecraft data

        plasma beta is calculated as ratio between thermal and magnetic pressure

        beta = 2*mu0*p / B^2

        where pressure, p, is assumed to be isotropic (p=nkT) for each species,
        mu0 is the permeability of free space, and B is the local magnetic field
        magnitude

        Args
        ----
        themis_df     pandas DataFrame saved from extracting data from CDF file


        Returns
        -------
        plasma_beta   pandas Series. index is a DatetimeIndex that matches the
                      themis_df. values are the calculated plasma beta for each
                      timestamp

        """
        from math import pi

        # Permeability of free space in SI units, H/m.
        mu0 = pi * 4e-7

        # Unit conversions:
        density_conv = 1e6 #cm^-3 to m^-3
        temp_conv = 1.602e-19 #eV to J
        mag_conv = 1e-9 #nT to T

        # Plasma beta assuming scalar temperature (and scalar pressure).
        Ni = themis_df.ionDensity * density_conv # m^-3
        Ne = themis_df.eDensity * density_conv
        Ti = themis_df.ionTemp * temp_conv # J
        Te = themis_df.eTemp * temp_conv

        # Ion and electron pressures assuming isotropic and equilibrium,
        # (not a necessarily valid assumption), p=nkT; temperature is in
        # units of energy already.
        ion_pressure = Ni * Ti
        elc_pressure = Ne * Te

        # We need to find the local magnetic field magnitude in order to
        # calculate the magnetic pressure.
        bx = themis_df.BX_DSL * mag_conv
        by = themis_df.BY_DSL * mag_conv
        bz = themis_df.BZ_DSL * mag_conv

        bmag_squared = bx**2 + by**2 + bz**2

        # This produces a Series of beta values matching the DatetimeIndex of
        # the themis_df.
        plasma_beta = 2 * mu0 * (ion_pressure + elc_pressure) / bmag_squared

        return plasma_beta


    def calc_mlt(self, x, y):
        """
        Returns the magnetic local time value given Geocentric Solar Magnetic
        *x* and *y* coordinates.
        """

        from numpy import arctan2, pi

        # Using the negative of (x, y) because of the way that the GSM
        # coordinate system is related to MLT, where midnight is at (-x, y=0)
        # and dawn (MLT 06) is at (x=0, -y).
        radians = arctan2(-y, -x)

        # The returned range of arctan2 is [-pi, pi); we want [0, 2pi).
        radians[radians<0] += 2*pi

        # There are 24 hours of MLT per 2pi, or 24MLT/2pi = 12/pi.
        return radians * 12 / pi


    def _calc_mlt_value(self, themis_df):
        """
        returns the mlt value, given a themis_df with a gsm longitude column


        """
        # Number of degrees offset from 00 MLT
        offset = 180

        # Conversion factor between longitude (in degrees) and MLT
        conversion = 15

        # Define a function to apply to the longitude values
        def mlt_value(longitude):
            if longitude >= offset:
                mltval = longitude - offset

            else:
                mltval = longitude + offset

            return mltval / conversion


        return themis_df.gsmLongitude.apply(mlt_value)



    def _calc_mlt_bin(self, themis_df):
        """
        calculates the MLT bin for spacecraft location based on GSM Longitude

        returns the calculated MLT bin values as a Series with same index as
        passed data frame


        Args
        ----
        themis_df         pandas DataFrame of themis data extracted from CDF


        Returns
        -------
        mlt_bins          pandas Series with MLT bins for each row of the
                          passed pandas DataFrame
        """
        mlt_bins = themis_df.gsmLongitude.apply(self._convert_long_mltbin)

        return mlt_bins



    def _convert_long_mltbin(self, longitude):
        """
        converts gsm longitude to magnetic local time position

        Args
        ----
        longitude    scalar; gsm longitude in degrees


        Returns
        -------
        mlt_bin      the bin size for the number of hours of mlt in each bin


        """
        # The following is for MLT bins that are 2 hours wide.
        if longitude >= 0 and longitude < 30:
            mlt_bin = 13
        elif longitude < 60:
            mlt_bin = 15
        elif longitude < 90:
            mlt_bin = 17
        elif longitude < 120:
            mlt_bin = 19
        elif longitude < 150:
            mlt_bin = 21
        elif longitude < 180:
            mlt_bin = 23
        elif longitude < 210:
            mlt_bin = 1
        elif longitude < 240:
            mlt_bin = 3
        elif longitude < 270:
            mlt_bin = 5
        elif longitude < 300:
            mlt_bin = 7
        elif longitude < 330:
            mlt_bin = 9
        elif longitude <= 360:
            mlt_bin = 11
        else:
            print('longitude out of range: setting MLT to -111')
            return -111

        return mlt_bin




    def segregate_by_mlt(self, data_fldr=THEMIS_DATA_FLDR):
        """
        Segregates combined THEMIS data frame into individual data frames
        for each MLT bin. Assumes that the ThemisData object was initialized
        with fname that contains the data to be segregated.

        Args:
        -----
        data_fldr   str; path to data folder where combined Themis DataFrame
                    is saved

        Returns:
        --------
        None        saves individual DataFrame objects to disk in folder
                    'data_fldr' as serialized pickle objects.

        """
        from pathlib import Path

        # Make boolean arrays for each MLT bin.
        unique_mlt_bins = self.data.mltBin.unique()

        # Take slices of the original DataFrame, one for each MLT bin.
        for b in unique_mlt_bins:
            # Get the filename; if it already exists, do not overwrite.
            mlt_slice_fname = self._make_mlt_segregate_fname(b)
            if Path(mlt_slice_fname).exists():
                continue

            # Otherwise get the data for this mlt bin.
            else:
                bool_mltbin = self.data.mltBin == b
                themis_by_mlt = self.data.loc[bool_mltbin, :]

                # Save the resulting DataFrame to disk.
                save_data(themis_by_mlt, mlt_slice_fname)

        return None


    def _make_mlt_segregate_fname(self, mltbin, data_dir=THEMIS_DATA_FLDR):
        """

        """
        from pathlib import Path

        fldr = data_dir + 'Mlt' + str(mltbin) + '/Rad' + self.radbin + '/'
        Path(fldr).mkdir(parents=True, exist_ok=True)

        fname = fldr + 'themis_mlt' + str(mltbin) + '_rad' + self.radbin \
              + '_' + self.begin_date + '_' + self.end_date + '.pkl'


        return fname








# --------------------------------------------------------------------------- #
class OmniData(TrainingData):
    """
    Contains methods and attributes for downloading, extracting, storing,
    and preparing OMNI data for neural network training

    """

    def __init__(self, day1=None, day2=None,
                 fname=None, single_event=False):
        """

        """
        #TODO Finish implementation of init.
        self.begin_date = day1
        self.end_date = day2
        self.single_event = single_event

        if fname is not None:
            self.data = super()._load_data(fname)

        else:
            self.data = None



    def get_begdate(self):
        """
        Returns the first date of OMNI data as a string in format "yyyymmdd"
        """
        return self.begin_date


    def get_enddate(self):
        """
        Returns the last date of OMNI data as a string in format "yyyymmdd"
        """
        return self.end_date


    def __str__(self):
        """

        """

    def __repr__(self):
        """

        """

    def download_cdf(self,
        dates,
        hro_res=ONE_MIN_OMNI,
        dnload_fldr=CDF_BASE_FLDR,
        date_freq='MS',
        ):
        """
        Downloads the requested CDF files from the NASA GSFC SPDF server
        if the file does not already exist either on the local machine,
        or on the CIFS server. If on the CIFS server, copies from CIFS;
        if on neither, downloads the CDF from NASA.

        Parameters:
        -----------
        dates : two-element tuple or list or single string
            string representations of the dates that are requested for the
            files in YYYYMMDD format. (e.g. ['20100101', '20100201'] will
            download one month worth of data, '20100101' will
            download a single day.)

        hro_res : string
            "hro_1min" or "hro_5min" for either 1-minute
            or 5-minute resolution data; Default is 1-minute

        dnload_fldr : string
            the start of the path to save the CDF files;
            Defaults to CDF_BASE_FLDR; recommend to not change.
            The CDF will be downloaded into a file organization system that
            mirrors the ftp server at NASA GSFC SPDF.

        date_freq : string
            pandas DateOffset string. default='MS', which means month start.
            For more info, see:
            pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

            Note: this is useful to change if requesting hourly OMNI, as those
            CDF files include 6 months of data, versus HRO OMNI, which are
            monthly CDF files.


        Returns
        -------
        None
            prints error message to stdout for any CDF files that were not
            able to download. Otherwise, saves CDF file to disk in
            *dnload_fldr* location.

        """
        from pandas import date_range

        # These do not depend on user input.
        # ---------------------------------#
        mission = 'omni'
        level = 'omni_cdaweb'
        # ---------------------------------#

        # Check whether a single date or multiple dates was passed.
        if len(dates) == 2:
            date_array = date_range(dates[0], dates[1], freq=date_freq)

        else:
            date = dates[:-2] + '01'
            date_array = date_range(date, date, freq=date_freq)


        for d in date_array:
            ymd, = get_dtstring(d, 'ymd')

            super().download_cdf(
                dnload_fldr,
                mission,
                level,
                hro_res,
                ymd,
                )

        return None


    def extract_data(self, dates,
                     save_fldr=OMNI_DATA_FLDR,
                     cdf_fldr=CDF_BASE_FLDR,
                     hro_res=ONE_MIN_OMNI,
                     version='omni',
                     date_freq='MS',
                     ):
        """
        Extracts data from CDF files and builds Pandas DataFrame objects.
        Saves the DataFrame to disk as a pickled python object.

        If the CDF file does not already exist on the local machine, then
        download_cdf() is called in an attempt to download.

        *** uses spacepy.pycdf module to open and read the CDF files ***

        Parameters:
        -----------
        dates : two-element tuple or list OR single string
            string representations of the dates that are requested for
            the files in YYYYMMDD format. (e.g. ['20100101', '20100201'] will
            extract two months worth of data, '20100101' will
            extract only the single month.)

        save_fldr : string
            the start of the path to save the pickle files;
            Defaults to OMNI_DATA_FLDR; recommend to not change.

        cdf_fldr : string
            the start of the path to look for the CDF files;
            Defaults to CDF_BASE_FLDR; recommend to not change.
            The CDF files are in a file organization that
            mirrors the ftp server at NASA GSFC SPDF.

        hro_res : string
            "hro_1min", "hro_5min", or "hourly"
            Default is 1-minute

        version : string
            which version of OMNI to look for (OMNI or OMNI-2) default: "omni"

        date_freq : string
            pandas DateOffset frequency string; default='MS'


        Returns
        -------
        None
            prints error message to stdout for any CDF files that
            were not able to open. Otherwise, saves pickle file
            to disk in save_fldr location.

        """

        from pathlib import Path
        from pandas import date_range

        # ------------------------------------------------------------------ #
        cdf_data_fldr = cdf_fldr + 'omni/omni_cdaweb/' + hro_res + '/'

        # Check whether a single date or multiple dates was passed.
        if len(dates) == 2:
            date_array = date_range(dates[0], dates[1], freq=date_freq)

        else:
            date = dates[:-2] + '01'
            date_array = date_range(date, date, freq=date_freq)


        # Build the pandas dataframes, using self.build_omni_df() method.
        for date in date_array:

            # Get string representations of the date.
            ymd_str, year_str = get_dtstring(date, 'ymd', 'year')

            # Append to the path for this year.
            omni_data_fldr = save_fldr + hro_res + '/' + year_str + '/'

            # Create the folder if it does not already exist.
            Path(omni_data_fldr).mkdir(parents=True, exist_ok=True)

            # Name the output file.
            omni_fname = omni_data_fldr + version + '_' \
                       + hro_res + '_' + ymd_str + '_df.pkl'

            # Check that the file does not already exist.
            if Path(omni_fname).exists():
                continue

            # Make sure the cdf file exists; if not, download it.
            # Account for differences in the hourly filenames.
            if hro_res == 'hourly':
                res = 'h0_mrg1hr'
            else:
                res = hro_res

            omni_cdf_file = cdf_data_fldr + year_str + '/' + version + '_' \
                          + res + '_' + ymd_str + '_v01.cdf'

            if not Path(omni_cdf_file).exists():
                self.download_cdf(
                    ymd_str,
                    hro_res=hro_res,
                    )

            omni_df = self._build_omni_df(omni_cdf_file)

            # _build_omni_df method returns False if there was an error.
            if omni_df is not False:
                save_data(omni_df, omni_fname)


    def _build_omni_df(self, omni_cdf_fname):
        """
        Opens OMNI cdf files and extracts specific variables to make a pandas
        DataFrame with a DatetimeIndex and column names given by either cdf
        variable names or derived from them in the case of multi-dimensional
        data.


        Inputs:
        -------
        omni_cdf_fname    str; the name, of the OMNI CDF file including full
                               path

        Returns:
        --------
        omni_df      a dataframe with solar wind and IMF parameters
        """

        import pandas as pd
        from spacepy import pycdf

        # Open omni cdf.
        try:
            omni = pycdf.CDF(omni_cdf_fname)

        except Exception as e:
            #print('Could not open', omni_cdf_fname)
            #print(e)
            return False

        # Create an empty dataframe with a DatetimeIndex.
        time = omni['Epoch'][...]
        time_index = pd.DatetimeIndex(time)

        omni_df = pd.DataFrame([], index=time_index)

        # Fill the dataFrame with data from the CDF file.
        # Note that X_gse == X_gsm; and BX_GSE is what is in the omni cdf.
        omni_df['BX_GSM'] = omni['BX_GSE'][...]
        omni_df['BY_GSM'] = omni['BY_GSM'][...]
        omni_df['BZ_GSM'] = omni['BZ_GSM'][...]
        omni_df['Tp'] = omni['T'][...]
        omni_df['Beta'] = omni['Beta'][...]
        omni_df['flow_pressure'] = omni['Pressure'][...]
        omni_df['Efield'] = omni['E'][...]

        # There are slight differences in available parameters for high res
        # OMNI versus hourly OMNI.
        if time_index.resolution == 'minute':
            omni_df['flow_speed'] = omni['flow_speed'][...]
            omni_df['Np'] = omni['proton_density'][...]
            omni_df['VX_GSE'] = omni['Vx'][...]
            omni_df['VY_GSE'] = omni['Vy'][...]
            omni_df['VZ_GSE'] = omni['Vz'][...]
            omni_df['BSNx_GSE'] = omni['BSN_x'][...]
            omni_df['BSNy_GSE'] = omni['BSN_y'][...]
            omni_df['BSNz_GSE'] = omni['BSN_z'][...]

        elif time_index.resolution == 'hour':
            omni_df['flow_speed'] = omni['V'][...]
            omni_df['Np'] = omni['N'][...]
            omni_df['Dst'] = omni['DST'][...]
            omni_df['Kp'] = omni['KP'][...]
            omni_df['F107'] = omni['F10_INDEX'][...]

        else:
            print("Time resolution of current OMNI data not implemented.")
            omni.close()
            return False

        # Close omni cdf file.
        omni.close()

        # In case any CDFs have duplicated entries in time array.
        if omni_df.index.has_duplicates:
            # Keep only the first occurrence of each time.
            omni_df = omni_df.loc[~omni_df.index.duplicated(), :]

        # Sorting index if needed, in case the index is not monotonic.
        if not omni_df.index.is_monotonic:
            omni_df.sort_index(inplace=True)

        return omni_df


    def normalize_omni(self, omni_data, omni_norm_stats):
        """
        return *omni_data* normalized by the values in *omni_norm_stats*

        The normalization procedure is to subtract the mean and divide by
        the standard deviation. *omni_norm_stats* is expected to be a
        dataframe with means and standard deviations along the index and having
        column names matching the column names of *omni_data*. The stats
        should be calculated from climatic periods, e.g. a solar cycle or
        longer.

        """

        from pandas import DataFrame

        # Create an empty DataFrame to be filled with normalized data.
        normed_omni = DataFrame([], columns=omni_data.columns)

        # Normalize each column and add to the normalized dataframe.
        for col in omni_data.columns:
            column_mean = omni_norm_stats.loc['mean', col]
            column_std = omni_norm_stats.loc['std', col]

            normed_omni[col] = \
                (omni_data.loc[:, col] - column_mean) / column_std


        return normed_omni


    def get_monthly_omni(self, year_mon, omni_dir=OMNI_DATA_FLDR):
        """
        return the OMNI DataFrame for the month *year_mon*

        The method first attempts to read the data saved locally from disk,
        if not present, then it attempts to download and extract CDF from
        SPDF.

        Args:
        -----
        year_mon    string; the year and month of the requested DataFrame
        omni_dir    string; the directory where the data frame is expected
                            to be stored. default=OMNI_DATA_FLDR.
                            recommend to not change.


        Returns
        -------
        omni_df     pandas DataFrame with OMNI data for the requested
                    month
        """

        from pathlib import Path
        # The OMNI dataframes are saved with a pattern filename.
        omni_fname = year_mon[:4] + '/omni_' + year_mon + '01_df.pkl'

        # Attempt to read the data frame if it is found; if not, then call
        # the extract_data method from self before reading file.
        if not Path(omni_dir + omni_fname).exists():
            ymd = year_mon + '01'
            self.extract_data(ymd)

        return super()._load_data(omni_dir + omni_fname)



    def handle_missing_omni(self, omni_df):
        """
        finds missing data in OMNI dataframe and converts the missing data into
        NaN values. Returns the modified dataframe
        see   https://omniweb.gsfc.nasa.gov/html/ow_data.html#1
        for information about missing values

        Args:
        -----
        omni_df      pandas DataFrame with OMNI data that was extracted from a
                     CDF file

        Returns:
        --------
        df_withNaNs  Original dataframe but with the missing data converted to
                     math.nan data type
        """

        from math import nan
        from numpy import isclose

        for column_name in OMNI_FILL_VALUES:

            # We have to use numpy.isclose method because there are errors
            # in the missing fill values; they are not exact sometimes.
            missing_bools = isclose(
                omni_df[column_name], 
                OMNI_FILL_VALUES[column_name],
                1e-1,
                )

            omni_df.loc[missing_bools, column_name] = nan


        return omni_df


    def interpolate_omni(self,
        data,
        method=None,
        axis=0,
        limit_minutes=54,
        ):
        """
        fills in missing data using interpolate method of *method*

        Parameters:
        -----------
        data : pandas dataframe or series
            array with missing data encoded as NaN

        method : str
            the method to use for interpolation; default is to
            not interpolate

        axis : int or str; (0, 1, "index" or "columns")
            the axis on which to operate over, default is 0

        limit_minutes : int
            the number of minutes for which the data can have consecutive
            missing entries and still perform interpolation
            default is 54


        Returns:
        --------
        interpolated_data : DataFrame with interpolated data
        """
        if method is not None:
            # Interpolate the data, as requested.
            interpolated_data = data.interpolate(method=method,
                                                 axis=axis,
                                                 limit=limit_minutes,
                                                 limit_area='inside',
                                                 limit_direction='forward')

            return interpolated_data

        else:
            # Do not interpolate; return original dataframe.
            return data


    def calc_malfven(self, V, B, n):
        """
        calculate and return the Alfven Mach number of the solar wind flow.

        Parameters
        ----------
        V : array-like or scalar, flow speed in km/s

        B : array-like or scalar, magnitude of IMF in nT

        n : array-like or scalar, number density of solar wind in cm^(-3)

        Note: V, B, n must have same shape.

        Returns
        -------
        Ma : same type, shape as V, B, n; Alfven Mach number, unitless
        """
        # Equation provided by OMNI; differs from Baumjohann and Treumann by
        # 10%, where their equation is 22 * B * n^(-1/2). This difference may
        # be attributable to the extra mass contributed by alpha particles in
        # the solar wind.
        Va = 20 * B / n**(1/2)    # result is in km/s

        return V / Va


    def calc_Palpha(self, Bx, By, Bz, V, N, alpha=0.45):
        """
        Calculate and return the coupling function labeled P_alpha in
        Equation 6 of Lockwood, et al. 2019. doi: 10.1029/2018SW001856.

        The default value for alpha is that which Lockwood et al., 2019 found
        to optimize the correlation coefficient between the coupling function
        and the am geomagnetic index for a period of 3 hours (see Table 2).

        alpha is the free parameter of Equation 6 that is found empirically.

        Note that in this calculation, I set all constants to unity. The
        parameter will be normalized before NN training, so it is not
        necessary to include constants here.

        Parameters
        ----------
        Bx : array-like, GSM-x IMF (nT)

        By : array-like, GSM-y IMF (nT)

        Bz : array-like, GSM-z IMF (nT)

        V : array-like, solar wind flow speed (km/s)

        N : array-like, solar wind proton density (cm^-3)

        alpha : scalar, correction to powers of the solar wind parameters
            used in coupling function. It is an empirically derived parameter
            to maximize correlation coefficient between coupling function
            and am geomagnetic index;
            the default value is that found by Lockwood to be optimal for the
            three hour period.

        Note: Bx, By, Bz, V, N must all have same shape.

        Returns
        -------
        Palpha : same type and shape as Bx, calculated Palpha with all
            constants set to unity.

        """
        from numpy import sin

        B = super().calc_vector_mag(Bx, By, Bz)
        theta = self.calc_clockangle(By, Bz)

        Palpha = B**(2*alpha) * V**((7/3)-2*alpha) * N**((2/3)-alpha) \
               * sin(theta/2)**4

        return Palpha


    def calc_Palpha1(self, Bx, By, Bz, V, N, alpha=0.38, fs=0.74):
        """
        From Lockwood 2019. doi:10.1029/2019JA026639
        Palpha1 includes energy input from Poynting flux vector
        (electromagnetic energy) in addition to Palpha, which the latter
        includes particle kinetic energy only.

        The calculation here makes the assumption/simplification that V
        is in negative GSM-x direction only.

        The values of the default parameters alpha and fs are taken from
        Lockwood 2019 as the values that provide the best correlation with
        the am geomagnetic index.

        Note that in this calculation, I set all constants to unity. The
        parameter will be normalized before NN training, so it is not
        necessary to include constants here.

        Parameters
        ----------
        Bx : array-like, GSM-x IMF (nT)

        By : array-like, GSM-y IMF (nT)

        Bz : array-like, GSM-z IMF (nT)

        V : array-like, solar wind flow speed (km/s)

        N : array-like, solar wind proton density (cm^-3)

        alpha : scalar, correction to powers of the solar wind parameters
            used in coupling function. It is an empirically derived parameter
            to maximize correlation coefficient between coupling function
            and am geomagnetic index;
            the default value is that found by Lockwood to be optimal

        fs : scalar, fraction of total solar wind Poynting flux that enters
            the magnetosphere;
            the default value is that found by Lockwood to optimize the CC
        """
        from numpy import cos

        # Equation 8 from Lockwood 2019.
        Palpha = self.calc_Palpha(Bx, By, Bz, V, N, alpha)

        # Need B to calculate Ma.
        B = super().calc_vector_mag(Bx, By, Bz)

        # Alfven Mach number.
        Ma = self.calc_malfven(V, B, N)

        # The angle between the IMF and flow speed, assuming all of flow speed
        # is in the -X GSM direction.
        phi = self.calc_angle_between_vectors(Bx, By, Bz, -1, 0, 0)

        # Inferred from Equation 9 of Lockwood 2019.
        psi = 2 * fs * cos(phi)**2 * Ma**(2*alpha - 2)

        # According to Equation 9 of Lockwood 2019.
        return Palpha * (1 + psi)


    def calc_angle_between_vectors(self, Bx, By, Bz, Vx, Vy, Vz):
        """
        calculates and returns the angle (in radians) between two vectors
        in three dimensions, given the six components of the vectors.

        Parameters
        ----------
        Bx, By, Bz : array-like or scalar, the first three components of the
            first vector

        Vx, Vy, Vz : array-like or scalar, the first three components of the
            second vector

        Returns
        -------
        phi : scalar, the angle (in radians) between the two vectors
        """
        from numpy import arccos

        # Calculate the vector magnitudes.
        B = super().calc_vector_mag(Bx, By, Bz)
        V = super().calc_vector_mag(Vx, Vy, Vz)

        # Calculate unit dot product.
        dot_product = Bx * Vx + By * Vy + Bz * Vz
        unit_dot_product = dot_product / (B * V)

        return arccos(unit_dot_product)


    def calc_bs(self, Bz):
        """
        calculate and return the rectified IMF Bz component, defined as
        Bs = Bz if Bz < 0, otherwise Bs = 0.

        Parameters
        ----------
        Bz : Series, IMF Bz in GSM coordinates

        Returns
        -------
        Bs : Series, same shape as Bz, with only non-positive values
        """

        Bs = Bz.copy()

        # Create a boolean array for positive values in Bz.
        positive_values = Bz > 0

        # Use boolean array to set positive values to 0.
        Bs.loc[positive_values] = 0

        return Bs


    def calc_geoeffective_efield(self, Bz, Vx):
        """
        calculate and return the geoeffective electric field defined as
        E_eff = -VBs, where
        V = solar wind velocity in GSM-x direction,
        Bs = rectified IMF Bz, such that Bs = Bz if Bz < 0, and 0 otherwise.

        Parameters
        ----------
        Bz : array-like or scalar, same shape as Vx, IMF in GSM-z
            direction in nanoTesla (nT)

        Vx : array-like or scalar, same shape as Bz, solar wind velocity
            in GSM-x direction in kilometers per second (km/s)

        Returns
        -------
        E_eff : array, same shape as Bz and Vx, the geoeffective electric
            field, in milliVolts per meter (mV/m)
        """

        unit_conversion_factor = 1e-3
        Bs = self.calc_bs(Bz)

        return - Vx * Bs * unit_conversion_factor


    def calc_clockangle(self, By, Bz):
        """
        calculate and return the clock angle between the interplanetary
        magnetic field (IMF) and the Northward direction in the
        Geocentric Solar Magnetic (GSM) coordinate system.

        Computes and returns:

        angle = arctan(By / Bz)

        using numpy.arctan2(), which returns the angle in the correct
        quadrant, and not restricted to the domain (-pi/2, pi/2).

        Parameters
        ----------
        By : array-like or scalar, the component of the IMF in the GSM-y
            direction; must be same shape as Bz

        Bz : array-like or scalar, the component of the IMF in the GSM-z
            direction; must be same shape as By

        Returns
        -------
        angle : same type as Bz and By, the calculated angle in radians of
            the direction of the IMF in the GSM yz-plane.

        By, Bz must be given in same units, typically this will be nanoTesla.
        """

        from numpy import arctan

        return arctan(By / Bz)


    def calc_narmax_cf(self, p, V, By, Bz):
        """
        calculates and returns the coupling function defined in
        Boynton et al., 2011 (doi: ) where the NARMAX algorithm was used to
        deduce the most appropriate power to each solar wind parameter.

        The optimal coupling function takes the form:

        p^(1/2) * V^(4/3) * B_T + sin(theta/2)^6,

        where p is dynamic pressure, V is flow speed, B_T is tangential IMF,
        B_T = (By^2 + Bz^2)^(1/2), theta is the IMF clock angle.
        Here the simplification and approximation will be used that the
        flow direction is only in the GSM-x direction, so that B_T is
        the strength of the magnetic field in the GSM yz-plane.

        Parameters
        ----------
        p : Series, solar wind flow pressure in nPa

        V : Series, solar wind flow speed in km/s

        By : Series, IMF GSM-y direction in nT

        Bz : Series, IMF GSM-z direction in nT

        Returns
        -------
        cf : Series, the calculated coupling function in arbitrary units
        """

        from numpy import sin

        # Get the clock angles.
        theta = self.calc_clockangle(By, Bz)

        # Get the tangential magnetic field.
        Bt = super().calc_vector_mag(By, Bz)

        return p**(1/2) * V**(4/3) * Bt * sin(theta/2)**6


    def calc_ulf_wave_power(self,
                            signal,
                            interp_method='akima',
                            back_win_size=60,
                            fft_win_size=30,
                            dt=60,
                            window_shift=1,
                            freq_range=[0.001666, 0.006667],
                            fft_norm='ortho',
                            show_remaining_time=True):
        """Calculates ULF wave power following the methods outlined in
        Wang et al., 2017. doi:10.1002/2016JA023746.

        Parameters
        ----------
        signal : Series, the column of the solar wind parameter upon
            which the calculation will be performed; It is assumed that
            missing data are represented by NaNs.

        interp_method : str, the interpolation method that will be passed
            to numpy.interp1d function. See numpy documentation for
            possible values and their meaning. Default is to use the
            Akima piecewise cubic spline method

        back_win_size : int, the number of points to use in subtracting
            background; default is 60 (one hour)

        fft_win_size : int
            the number of points to use in the FFT moving window;
            default is 30 (thirty minutes); use only even number, not odd.

        dt : scalar, the step size in seconds of each observation.
             For HRO-1min, this is 60 seconds.

        window_shift : int, the number of points to move successive windows
            between each Fourier transform, default is 1, which is to not
            skip any time steps.

        freq_range : two-element 1d array-like
            the range of frequencies (in Hertz) to use when calculating the
            wave power index. Default is [0.001667, 0.006667] (Pc5 band).

        fft_norm : str or None, how to perform the normalization of the
            Fourier coefficients, default is 'ortho' which is to normalize
            both the forward and inverse transform by 1/sqrt(n).

        Returns
        -------
        signal_power : Series, same length and index as signal, calculated
            ULF wave power


        The steps below are used to calculate ULF wave power (note that these
        are modified from the steps outlined by Wang et al., 2017).

        1) Fill missing data with a Akima cubic interpolation. (This differs
            from Wang et al., 2017. They used a cubic spline interpolation.)

        2) Subtract background by using a smoothing function with a window
           of 60 points.

        3) Use FFT with 64 point (1h) moving window (1 min time shift
           between successive windows) to produce the power spectra
           in the ULF Pc5 frequency range.

           3.1) A Hann window is used to minimize edge effects that
                result in spectral leakage.

                The Hann window is defined as w(n) = 0.5-0.5*cos(2pin/(M-1))
                where 0 <= n <= M-1 and M is the number of points used and
                w(n) is the value of the Hann window at the n-th point.

        4) Compute ULF power index, which is defined as the sum of the
           power spectra over frequencies 0.5-8.3 mHz.

        """
        import datetime
        from scipy.signal.windows import hann
        from scipy import fft
        from numpy import absolute
        from pandas import DataFrame


        # 1) Interpolate the data.
        signal.interpolate(method='akima', inplace=True)


        # 2) Subtract background.
        signal = signal - signal.rolling(window=back_win_size,
                                         center=True,
                                         closed='both',
                                         min_periods=1).mean()

        # 3) Calculate FFT on a rolling window.
        wave_spectra = dict()
        hann_window = hann(M=fft_win_size, sym=False)
        #TODO determine if line is used: df = 1 / (fft_win_size * dt)

        # Use only the returned positive frequencies.
        freqs = absolute(fft.fftfreq(fft_win_size, dt)[:fft_win_size//2 + 1])

        loop_begin_time = datetime.datetime.now()
        num_steps = len(signal.index) - fft_win_size
        query_step_size = num_steps // 10
        for i in range(0, num_steps, window_shift):
            if show_remaining_time:
                if i == 100:
                    display_remaining_time(
                        i,
                        num_steps,
                        window_shift,
                        loop_begin_time)

                if i % query_step_size*window_shift == 0 and i > 0:
                    display_remaining_time(
                        i,
                        num_steps,
                        window_shift,
                        loop_begin_time)

            # Calculate the window of size N. Use a Hann window for shaping.
            window = signal.iloc[i:i+fft_win_size] * hann_window

            # Use the center of the window as the row label.
            row_label = window.index[fft_win_size//2 - 1]

            # Calculate FFT coefficients and the spectral power.
            # Setting overwrite_x=True discards the memory space taken by
            # window.values
            coeffs = fft.rfft(window.values, norm=fft_norm, overwrite_x=True)
            wave_spectra[row_label] = absolute(coeffs)**2

        # Create a DataFrame from the dictionary.
        wave_spectra = DataFrame.from_dict(wave_spectra,
                                           orient='index',
                                           columns=freqs,
                                           dtype=float)

        # Calculate the wave power index using only the frequencies between
        # 0.5 - 8.3 mHz.
        indices = freqs.searchsorted(freq_range)
        wave_power = wave_spectra.iloc[:, indices[0]:indices[1]].sum(axis=1)

        # Return both the calculated wave power and the full wave spectra.
        return wave_power, wave_spectra


    def concatenate_omni(self,
        data_dir,
        save_dir,
        start_month,
        end_month,
        version='omni',
        resolution='hro_1min',
        ):
        """
        Concatenates and creates rolling calculated values for OMNI data
        that is stored in monthly dataframes. Also saves a concatenated
        file without performing any calculations. The original data has
        filename with "all" appended to it.

        Parameters
        ----------
        data_dir : str
            the directory where the monthly OMNI dataframes are located

        save_dir : str
            the directory to where the rolling, calculated
            parameters should be stored

        start_month : str or datetime-like
            if str use ISO format; the starting month. example: "20080201"

        end_month : str or datetime-like
            if str use ISO format; the ending month. example: "20090201"

        version : str, default 'omni'
            either "omni" or "omni2"; which version of OMNI data to use

        resolution : str, default 'hro_1min'
            the resolution of OMNI; e.g. 'hro_1min' for high resolution OMNI
            at 1 minute


        Returns: pandas DataFrame
            the concatenated dataframe
            Also saves the resulting DataFrame to disk.

        """
        from pathlib import Path
        from pandas import date_range, read_pickle, concat

        # Create an array of dates that start on the first day of each month.
        all_months = date_range(start_month, end_month, freq='MS')

        # It is more efficient to concatenate a list of DataFrames
        # than to use the append method of a DataFrame.
        monthly_omni_dfs = list()
        for month in all_months:
            ymd = month.strftime('%Y%m%d')
            year = ymd[:4]
            month_fname = data_dir + year + '/' + version \
                        + '_' + resolution + '_' + ymd + '_df.pkl'
            monthly_omni_dfs.append(read_pickle(month_fname))

        all_data = concat(monthly_omni_dfs, axis=0)
        all_data = self.handle_missing_omni(all_data)

        # Check no dates overlap.
        if all_data.index.has_duplicates:
            print("some dates overlap")

        base_fname = save_dir + version + '_' + start_month + '_' + end_month

        # Create the directory if it does not already exist.
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save the data as concatenated before calculating rolling stats.
        all_data.to_pickle(base_fname + '.pkl')

        return all_data









