# ----------------------------------------------------------------------- #
# Written By: Brian Swiger
# Purpose: define constants to use with NN Model Classes.
# ----------------------------------------------------------------------- #

# pseudo-constants
BASE_DATA_FLDR = './Data/'
THEMIS_DATA_FLDR = BASE_DATA_FLDR + 'Themis/'
OMNI_DATA_FLDR = BASE_DATA_FLDR + 'Omni/'
FISM_DATA_FLDR = BASE_DATA_FLDR + 'Fism/'
TRAINING_DATA_FLDR = BASE_DATA_FLDR + 'Training/'
MODEL_DATA_FLDR = BASE_DATA_FLDR + 'Model/'
CDF_BASE_FLDR = BASE_DATA_FLDR + 'CDFdata/'
FIGURE_FLDR = './Figures/'
BASE_URL = 'https://spdf.gsfc.nasa.gov/pub/data/'
ONE_MIN_OMNI = 'hro_1min'
MLT_BINS = [1, 3, 5, 19, 21, 23]
GOOD_FLAGS = [0, 8, 256, 264]
R_MIN = 6 #Earth radii
R_MAX = 12 #Earth radii

# Plasma beta; >1 indicates measurement is in plasma sheet.
# See Burin des Roziers et al., 2009; J. Geophys. Res. Space Physics
BETA_CUTOFF = 1

# These criteria limit the measurements to the nightside.
# GSM Longitude is 0 degrees at local noon, 90 degrees at local dusk,
# 180 degrees at midnight, and 270 degrees at dawn.
GSM_LONG_MIN = 90 #degrees
GSM_LONG_MAX = 270 #degrees

# Themis fill values for all parameters.
THEMIS_FILL_VALUES = {
                      'GMOM' : -1e30,
                      'SSC' : -1e31,
                      'FIT' : 'nan'
                      }


# OMNI missing data keys for each variable in the OMNI dataframe;
# See OMNI CDF variable attributes (metadata).
OMNI_FILL_VALUES = {
                    'BX_GSM' : 9999.99,
                    'BY_GSM' : 9999.99,
                    'BZ_GSM' : 9999.99,
                    'VX_GSE' : 99999.9,
                    'VY_GSE' : 99999.9,
                    'VZ_GSE' : 99999.9,
                    'Np' : 999.99,
                    'Tp' : 9999999.0,
                    'Beta' : 999.99,
                    'flow_pressure' : 99.99,
                    'flow_speed' : 99999.9,
                    'Efield' : 999.99,
                    'BSNx_GSE' : 9999.99,
                    'BSNy_GSE' : 9999.99,
                    'BSNz_GSE' : 9999.99
                   }




NNMODEL_INIT_ERROR_MSG = "Did not initialize with valid constructor. \n" \
                       + "The initialization must specify: \n"\
                       + "begin, end, as strings in yyyymmdd format.\n"

# ------------------------------------------------------------------------ #
# Creates a dictionary mapping index to channel energy (in keV).
_channels = ['1.300', '2.230', '3.750', '6.700', '11.65',
            '20.20', '27.00', '28.00', '29.00', '30.00',
            '31.00', '41.00', '52.00', '65.50', '93.00',
            '139.0', '203.5']

ENERGY_DICT = dict(zip(range(len(_channels)), _channels))

THA_CHNLS = ['7.3', '9.2', '12.6', '16.0', '20.8', '28.1', '36.8', '48.5',
             '63.5', '83.9', '110.5', '144.9', '190.5', '251.1', '330.1',
             '434.8', '571.5', '752.8', '991.8', '1304.5', '1717.5', '2260.9',
             '2976.8', '3917.7', '5157.3', '6788.9', '8936.4', '11763.4',
             '15484.9', '20383.3', '26831.4', '27000.0', '28000.0',
             '29000.0', '30000.0', '31000.0', '41000.0', '52000.0', '65500.0',
             '93000.0', '139000.0', '203500.0', '293000.0', '408000.0',
             '561500.0', '719500.0']

THC_CHNLS = ['7.2', '9.2', '12.5', '15.9', '20.7', '27.9', '36.6', '48.2',
             '63.1', '83.3', '109.8', '144.0', '189.3', '249.5', '328.1',
             '432.1', '568.0', '748.2', '985.7', '1296.4', '1706.8', '2246.9',
             '2958.4', '3893.5', '5125.3', '6746.9', '8881.1', '11690.6',
             '15389.0', '20257.1', '26665.3', '27000.0', '28000.0',
             '29000.0', '30000.0', '31000.0', '41000.0', '52000.0', '65500.0',
             '93000.0', '139000.0', '203500.0', '293000.0', '408000.0',
             '561500.0', '719500.0']

THD_CHNLS = ['7.1', '9.0', '12.4', '15.7', '20.5', '27.6', '36.2', '47.6',
             '62.4', '82.4', '108.6', '142.4', '187.2', '246.7', '324.3',
             '427.2', '561.5', '739.6', '974.4', '1281.6', '1687.4', '2221.2',
             '2924.7', '3849.1', '5066.8', '6669.9', '8779.7', '11557.2',
             '15213.3', '20025.8', '26360.9', '27000.0', '28000.0',
             '29000.0', '30000.0', '31000.0', '41000.0', '52000.0', '65500.0',
             '93000.0', '139000.0', '203500.0', '293000.0', '408000.0',
             '561500.0', '719500.0']

THE_CHNLS = ['7.2', '9.1', '12.4', '15.7', '20.5', '27.7', '36.2', '47.7',
             '62.5', '82.5', '108.7', '142.5', '187.4', '247.0', '324.7',
             '427.6', '562.1', '740.4', '975.4', '1282.9', '1689.1', '2223.6',
             '2927.7', '3853.1', '5072.1', '6676.9', '8788.9', '11569.3',
             '15229.3', '20046.9', '26388.6', '27000.0', '28000.0',
             '29000.0', '30000.0', '31000.0', '41000.0', '52000.0', '65500.0',
             '93000.0', '139000.0', '203500.0', '293000.0', '408000.0',
             '561500.0', '719500.0']


USE_CHNLS = ['82.5', '108.7', '142.5', '187.4', '247.0', '324.7',
             '427.6', '562.1', '740.4', '975.4', '1282.9', '1689.1', '2223.6',
             '2927.7', '3853.1', '5072.1', '6676.9', '8788.9', '11569.3',
             '15229.3', '20046.9', '26388.6', '27000.0', '28000.0',
             '29000.0', '30000.0', '31000.0', '41000.0', '52000.0', '65500.0',
             '93000.0', '139000.0', '203500.0']

# Removing energy channels >100keV.
TRAIN_CHNLS = USE_CHNLS[:-2]

ENERGY_CHNLS = {
                'tha' : THA_CHNLS,
                'thc' : THC_CHNLS,
                'thd' : THD_CHNLS,
                'the' : THE_CHNLS,
                'use' : USE_CHNLS
               }

LOW_ENERGY_GROUP = ['562.1', '740.4', '975.4']

MEDIUM_ENERGY_GROUP = ['1282.9', '1689.1', '2223.6', '2927.7', '3853.1',
                       '5072.1', '6676.9', '8788.9']

HIGH_ENERGY_GROUP = ['11569.3', '15229.3', '20046.9', '26388.6', '27000.0',
                     '28000.0', '29000.0', '30000.0', '31000.0', '41000.0']


