# Functions for calculating statistics / metrics.

# -----------------------
# Fit-performance metrics
# -----------------------

def calc_Q(y, y_hat):
    """
    calculates and returns Q, the ratio y_hat / y, which is predicted
    over observed

    y, y_hat       observed, predicted values; array or array like
    """
    return y_hat / y


def calc_MSA(y, y_hat):
    """
    calculates and returns Median Symmetric Accuracy, given observed *y* and
    predicted *y_hat* arrays.

    """
    from numpy import exp, median, log, absolute

    Q = calc_Q(y, y_hat)

    return 100 * (exp(median(absolute(log(Q)))) - 1.0)


def MSA_with_log10(y, y_hat):
    """
    calculates and returns Median Symmetrics Accuracy, given observed *y* and
    predicted *y_hat* arrays where the values are in log10 units.

    """
    from numpy import median, absolute, power

    # Median absolute log of Q, the y_hat / y ratio. Where Q is in units of
    # flux, and this function expects y, y_hat to be in units of log10 flux.
    malogQ = median(absolute(y_hat - y))

    return 100 * (power(10, malogQ) - 1.0)


def calc_MdLQ(y, y_hat):
    """
    calculates and returns Median Log Q, given observations, *y*, and
    predictions, *y_hat*.

    """
    from numpy import median, log

    Q = calc_Q(y, y_hat)

    return median(log(Q))


def calc_SSPB(y, y_hat):
    """
    calculates and returns Symmetric Signed Percentage Bias, given a set of
    observations, *y*, and predictions, *y_hat*.

    """
    from numpy import exp, sign, absolute

    MdLQ = calc_MdLQ(y, y_hat)

    return 100 * sign(MdLQ) * (exp(absolute(MdLQ)) - 1.0)


def calc_MSE(y, y_hat):
    """
    return mean squared error
    """
    from numpy import mean, square, subtract

    return mean(square(subtract(y, y_hat)))


def calc_RMSE(y, y_hat):
    """
    returns root mean squared error
    """
    from numpy import sqrt

    mse = calc_MSE(y, y_hat)

    return sqrt(mse)


def calc_MAE(y, y_hat):
    """
    returns mean absolute error
    """
    from numpy import mean, absolute, subtract

    return mean(absolute(subtract(y, y_hat)))


def calc_ME(y, y_hat):
    """
    returns Mean Error (bias)
    """
    from numpy import mean

    return mean(y_hat) - mean(y)


def calc_gME(y, y_hat):
    """
    returns the geometric mean error (geometric bias)
    """
    from scipy.stats.mstats import gmean

    return gmean(y_hat) - gmean(y)


def calc_var(y):
    """
    Returns the variance of a flattened (if *y* is not already 1d) array.
    """
    from numpy import var

    return var(y)


def calc_PE(y, y_hat):
    """
    returns prediction efficiency
    """
    variance = calc_var(y)
    mse = calc_MSE(y, y_hat)

    return 1 - mse / variance


def calc_SSmse(y, y_hat, y_hat_ref):
    """
    returns skill score based on MSE of a reference model.
    That is: 1 - (MSE / MSE_ref).
    """
    mse = calc_MSE(y, y_hat)
    mse_ref = calc_MSE(y, y_hat_ref)

    return 1 - mse / mse_ref


def calc_R(y, y_hat):
    """
    returns pearson correlation coefficient
    """
    from scipy.stats import pearsonr

    return pearsonr(y, y_hat)


def calc_MAPE(y, y_hat):
    """
    returns Mean Absolute Percentage Error
    """
    from numpy import mean, absolute

    return 100 * mean(absolute(y - y_hat) / y)


def calc_sMAPE(y, y_hat):
    """
    return symmetric MAPE
    """
    from numpy import mean, absolute

    return 200 * mean(absolute(y - y_hat) / absolute(y + y_hat))


def frac_within(
    y,
    yhat,
    delta=2,
    logged=True,
    ):
    """
    Returns the fraction of *yhat* points
    that are within a factor of *delta* from *y*.

    Parameters:
    -----------
    y : float or array-like, same shape as *yhat*
        the observed value(s)

    yhat : float or array-like, same shape as *y*
        the modeled values

    delta : scalar, default=2
        the amount that the data can be different from the observed and still
        be considered "within" bounds

    logged : bool, default=True
        whether the *y* and *yhat* data are in log units

    Returns:
    --------
    ip : scalar
        the fraction within the limit as a percentage

    """
    from numpy import absolute, log10, maximum, mean

    if logged:
        included = absolute(yhat - y) <= log10(delta)

    else:
        included = maximum(yhat/y, y/yhat) <= delta

    percent_included = mean(included) * 100

    return percent_included



def combine_fit_metrics(
        energy_channels,
        obsv_df,
        pred_df,
        include_SSmse=False,
        ref_pred_df=None,
        ):
    """
    Calculates several fit performance metrics and combines them into a
    DataFrame, ready for exporting to csv or similar.

    Parameters:
    -----------
    energy_channels : dict
        a dictionary whose keys and values describe energy channels

    obsv_df : pandas.DataFrame
        the observed flux in log10 format

    pred_df : pandas.DataFrame
        the predicted flux in log10 format

    include_SSmse : bool, default=False
        Whether to include the skill score based on MSE, which is useful for
        doing a model-model comparison.

    ref_pred_df : pandas.DataFrame, default None
        predicted flux of a reference model

        This must not be None if include_SSmse is True. include_SSmse requires
        the values from the reference model in its calculation.

    Returns:
    --------
    metrics_df : pandas.DataFrame
        A DataFrame of metrics calculated by energy channel.
    """
    from pandas import concat, DataFrame
    from util import unlog

    all_metrics = list()

    for channel_label in energy_channels.keys():

        # Work with only this channel or group of channels.
        channel = energy_channels[channel_label]

        obsv_logflux = obsv_df.loc[:, channel].values
        pred_logflux = pred_df.loc[:, channel].values

        # Return logged flux values back to unlogged flux values.
        obsv_flux = unlog(obsv_logflux)
        pred_flux = unlog(pred_logflux)

        pears_r, _ = calc_R(obsv_flux.flatten(), pred_flux.flatten())
        pears_r_log, _ = calc_R(obsv_logflux.flatten(),
            pred_logflux.flatten())

        # Add metrics as desired.
        metrics = {'R_log' : pears_r_log,
                   'MAPE_log' : calc_MAPE(obsv_logflux, pred_logflux),
                   'sMAPE_log' : calc_sMAPE(obsv_logflux, pred_logflux),
                   'PE_log' : calc_PE(obsv_logflux, pred_logflux),
                   'MSE_log' : calc_MSE(obsv_logflux, pred_logflux),
                   'MAE_log' : calc_MAE(obsv_logflux, pred_logflux),
                   'ME_log' : calc_ME(obsv_logflux, pred_logflux),
                   'MSA' : calc_MSA(obsv_flux, pred_flux),
                   'SSPB' : calc_SSPB(obsv_flux, pred_flux),
                   'IP' : frac_within(obsv_logflux, pred_logflux),
                   'obs_var_log' : calc_var(obsv_logflux),
                   'pred_var_log' : calc_var(pred_logflux),
                   'obs_var' : calc_var(obsv_flux),
                   'pred_var' : calc_var(pred_flux)
                  }

        if include_SSmse:
            ref_pred_logflux = ref_pred_df.loc[:, channel].values
            ssmse = calc_SSmse(
                obsv_logflux,
                ref_pred_logflux,
                pred_logflux,
                )
            metrics['SSmse'] = ssmse

        all_metrics.append(DataFrame(metrics, index=[channel_label]))

    metrics_df = concat(all_metrics)

    return metrics_df




# ------------------------
# Event-detection metrics.
# ------------------------
# TODO: add event detection metric calculations.














