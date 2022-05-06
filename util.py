# -------------------------------------------------------------------------- #
#
# Utility functions for pre- and post-processing of training and model data.
#
# Written By: Brian Swiger
# -------------------------------------------------------------------------- #


# =========================================================================== #
# Utility functions.
# =========================================================================== #
def convert_eflux_to_nflux(
    eflux_data,
    flux_log10=True):
    """
    Converts differential energy flux to differential number flux.

    Parameters
    ----------
    eflux_data : pandas DataFrame
        a dataframe of flux data with differential energy flux units in eV;
        eV/(cm^2 s sr eV) or log10(eV/(cm^2 s sr eV))
        The format of the DataFrame is: rows are observations, columns are
        channels with labels of the energy of the channel in eV.
        Example (with flux_log10=True):

                                 82.5      108.7    ...     203500.0
        ------------------------------------------------------------
        2007-12-01 00:02:00   6.662221  6.825370    ...     6.093056
        2007-12-01 00:03:00   5.016743  4.938895    ...     6.068558
                  :               :         :                   :
                  :               :         :                   :
                  :               :         :                   :
        2020-08-31 23:59:00   5.923173  5.921880    ...     4.505219


    flux_log10 : bool
        if True, the flux values in *flux_column* are the log base 10 of the
        actual flux. If False, the flux values represent actual flux.

    Returns
    -------
    flux_column : pandas Series
        a dataframe of flux data with differential number flux units in
        keV; 1/(cm^2 s sr keV) or log10(1/(cm^2 s sr keV))

    """
    from math import log10
    from numpy import ones
    from pandas import DataFrame

    nflux_data = DataFrame(
        ones(eflux_data.shape),
        index=eflux_data.index,
        columns=eflux_data.columns
        )

    for column_label in eflux_data.columns:
        energy = float(column_label)
        flux_column = eflux_data.loc[:, column_label]

        if flux_log10:
            nflux_data.loc[:, column_label] = flux_column + 3 - log10(energy)

        else:
            nflux_data.loc[:, column_label] =  flux_column * 1e3 / energy

    return nflux_data


def convert_nflux_to_eflux(
    nflux_data,
    flux_log10=True,
    ):
    """
    Convert differential number flux to differential energy flux.
    Original units:
        1/(cm^2 s sr keV)

    Returned units:
        eV/(cm^2 s sr eV)

    Note that in addition to the conversion, we are changing from keV to eV.

    Parameters:
    ----------
    nflux_data : pandas DataFrame
        a dataframe of flux data with differential number flux units in keV;
        1/(cm^2 s sr keV) or log10(1/(cm^2 s sr keV))
        The format of the DataFrame is: rows are observations, columns are
        channels with labels of the energy of the channel in eV.
        Example (with flux_log10=True):

                                 82.5      108.7    ...     203500.0
        ------------------------------------------------------------
        2007-12-01 00:02:00   6.662221  6.825370    ...     6.093056
        2007-12-01 00:03:00   5.016743  4.938895    ...     6.068558
                  :               :         :                   :
                  :               :         :                   :
                  :               :         :                   :
        2020-08-31 23:59:00   5.923173  5.921880    ...     4.505219


    flux_log10 : bool, default True
        if True, the flux values in *flux_column* are the log base 10 of the
        actual flux. If False, the flux values represent actual flux.

    Returns
    -------
    eflux_data : pandas Series
        a dataframe of flux data with differential energy flux units in
        eV; eV/(cm^2 s sr eV) or log10(eV/(cm^2 s sr eV))
        Data are returned in same logged format as provided: either logged or
        unlogged.

    """
    from math import log10
    from numpy import ones
    from pandas import DataFrame

    eflux_data = DataFrame(
        ones(nflux_data.shape),
        index=nflux_data.index,
        columns=nflux_data.columns
        )

    for column_label in nflux_data.columns:
        energy = float(column_label)
        flux_column = nflux_data.loc[:, column_label]

        if flux_log10:
            eflux_data.loc[:, column_label] = flux_column - 3 + log10(energy)

        else:
            eflux_data.loc[:, column_label] =  flux_column / 1e3 * energy


    return eflux_data













def unlog(x, base=10):
    """
    perform element-wise operation to return the unlogged values of *x*,
    whose logarithms were taken with base *base*.

    """
    from numpy import power

    return power(base, x)


def spatial_subsets(
    target_data,
    locations,
    save_dir,
    spatial_bins=[(-6, -2, 2, 6), (6, 9, 12)],
    ):
    """
    Given *target_data* and *locations* of the targets in the model domain
    (6-12Re, 18-06MLT), subset the data into *spatial_bins*.

    Parameters:
    -----------
    target_data : pandas DataFrame
        dataframe with fluxdata and DatetimeIndex; this can be test, train,
        validation targets or model output.

    locations: pandas DataFrame
        dataframe with columns of ['mlt', 'rdist'] having same DatetimeIndex
        as *target_data*

    save_dir : str
        which directory to save the subset dataframes

    spatial_bins : list of tuples

        the first tuple has the boundaries of the bins in the MLT direction,
        in hours relative to midnight

        the second tuple has the boundaries of the radial distance direction,
        in Earth radii.

        Default: [(-6, -2, 2, 6), (6, 9, 12)]


    Returns:
    --------
    None
        saves the subset dataframes to disk at *save_dir* with filename
        format 'save_dir/subset_mltM_rdistR.pkl'

        where M is the lower mlt bound and R is the lower radial distance bound
        for each bin.

        Ex: the subset for spatial bin (-2 < mlt < 2), (9 < rdist < 12) has
        filename 'save_dir/subset_mlt-2_rdist9.pkl'

    """

    # Get the requested bin boundaries.
    mlt_bins = spatial_bins[0]
    rdist_bins = spatial_bins[1]

    # Get the rdist and mlt coordinates.
    rdist_locs = locations.rdist
    mlt_locs = locations.mlt

    # Convert mlts on duskside to relative to midnight, i.e. 18-24 --> -6 to 0.
    mlt_locs[mlt_locs >= 18] -= 24

    for i in range(len(rdist_bins) - 1):
        # Create rdist boolean.
        rdist_lower = rdist_bins[i]
        rdist_upper = rdist_bins[i+1]

        rdist_subset = (rdist_locs > rdist_lower) & (rdist_locs <= rdist_upper)

        for j in range(len(mlt_bins) - 1):
            # Create mlt boolean.
            mlt_lower = mlt_bins[j]
            mlt_upper = mlt_bins[j+1]

            mlt_subset = (mlt_locs > mlt_lower) & (mlt_locs <= mlt_upper)

            # Combine booleans.
            loc_subset = rdist_subset & mlt_subset

            # Subset fluxdata by spatial boolean.
            subset = target_data.loc[loc_subset, :]

            # Define filename.
            save_fname = save_dir + \
                '/subset_mlt' + \
                str(mlt_lower) + \
                '_rdist' + \
                str(rdist_lower) + \
                '.pkl'

            # Save to disk.
            subset.to_pickle(save_fname)


    return None


def integrate_flux_values(
        data,
        emin,
        emax,
        log10_flux=True
        ):
    """
    Integrate flux values in *data* over a range of energies
    from *emin* to *emax*.
    This is really a weighted average of the flux in energy channels between
    emin and emax, weighted by the width of each energy channel.

    Parameters:
    -----------
    data : pandas DataFrame
        dataframe of flux values with datetime index and column labels as
        strings for each energy channel in eV.
    emin : float
        energy in eV; the minimum energy of the range over which to
        integrate fluxes.

    emax : float
        energy in eV; the maximum energy of the range over which to
        integrate fluxes.

    log10_flux : bool, default=True
        If True, the values in *data* are the log base 10 of the actual flux
        values. The integrated flux will be returned in log base 10 values.

        If False, the values in *data* are the actual flux values observed and
        the integrated flux will be returned as the actual flux values.

        Note that the integration is calculated using actual flux values in
        either case. If log10_flux is True, then the values are unlogged,
        integrated, then log10 re-applied.

    Returns:
    --------
    j_dE : pandas Series
        column of integrated flux values for energy range dE = emax-emin.
    """
    from numpy import log10

    if log10_flux:
        data = unlog(data)

    # Create array of energy labels as floats.
    energy_labels = data.columns.astype(float)
    assert energy_labels.is_monotonic_increasing, \
        "Columns of dataframe are not sorted in monotonic increasing order."

    # Search through energy labels and find indices to place emin and emax.
    emin_idx = energy_labels.searchsorted(emin)
    emax_idx = energy_labels.searchsorted(emax)

    integrated_flux = 0.0

    for i in range(emin_idx, emax_idx-1):
       de = energy_labels[i+1] - energy_labels[i]
       integrated_flux += de * data.iloc[:, i]


    # Add the minimum energy width * flux.
    de_min = energy_labels[emin_idx] - emin
    if emin_idx != 0:
        integrated_flux += de_min * data.iloc[:, emin_idx-1]
    else:
        integrated_flux += de_min * data.iloc[:, emin_idx]

    # Add the maximum energy width * flux.
    de_max = emax - energy_labels[emax_idx-1]
    integrated_flux += de_max * data.iloc[:, emax_idx-1]

    # Calculate dE and divide integrated flux.
    integrated_flux /=  (emax - emin)

    # Name the Series object with the range of energy.
    integrated_flux.name = str(emin) + "-" + str(emax)


    if log10_flux:
        return log10(integrated_flux)

    else:
        return integrated_flux






def add_integrated_flux_channels(
    fluxdata,
    energy_ranges,
    save_fname,
    log10_flux=True,
    ):
    """
    Add columns of integrated flux from combinations of existing channels
    that are within the ranges specified in *energy_ranges*.
    Saves a new dataframe to *save_fname* as pickle object.
    The data in the new dataframe is in the same format as given in *fluxdata*
    (i.e., if data in *fluxdata* is log10,
    then data in new dataframe is in log10.)

    Parameters:
    -----------
    fluxdata : pandas DataFrame
        existing dataframe with columns of energy channels

    energy_ranges : list of two-element tuples of floats
        the energy ranges to create the integrated flux channels
        example: [(1e3, 1e4), (1e4, 1e5), ... , (1e2, 1e3)]

    save_fname : str
        filename with path of where to save the new dataframe

    log10_flux : bool; default True
        whether the data in *fluxdata* are in log10 format


    Return:
    -------
    None
        saves the new dataframe with added integrated flux columns to disk
        at *save_fname*

    """

    from pandas import concat, DataFrame

    # Create empty list to store the new columns temporarily.
    integrated_fluxes = []

    for energy_range in energy_ranges:
        # Pull the values from the ranges.
        min_energy = energy_range[0]
        max_energy = energy_range[1]

        # Calculated the integrated flux.
        integrated_fluxes.append(
            integrate_flux_values(
                data=fluxdata,
                emin=min_energy,
                emax=max_energy,
                log10_flux=log10_flux,
                )
            )

    # Create new dataframe of the new integrated channels.
    integrated_fluxes = concat(integrated_fluxes, axis=1)

    fluxdata_with_integrated = concat(
        [fluxdata, integrated_fluxes],
        axis=1,
        )

    # Save the new dataframe to disk.
    fluxdata_with_integrated.to_pickle(save_fname)

    return None





























































