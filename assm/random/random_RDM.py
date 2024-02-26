import numpy as np
import pandas as pd

def random_gard_2A(inc_drift, 
                   cor_drift, 
                  sigma, 
                  ndt, 
                  spvar=None,
                  starting_point_variability=False,
                  noise_constant=1, dt=0.001, max_rt=10):
   
    # Based on the Wiener diffusion process
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt


    if starting_point_variability:
        x_cor = np.random.uniform(0, spvar)
        x_inc = np.random.uniform(0, spvar)
    else:
        x_cor = np.zeros(shape)
        x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           sigma[ongoing] * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           sigma[ongoing] * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= 1)
        ended_incorrect = (x_inc >= 1)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc

def random_gard_nA(drift, sigma, ndt, dt=0.001, max_rt=10):
    shape = ndt.shape
    n_options = drift.shape[2]
    choice = np.empty(shape)*np.nan
    rt = np.empty(shape)*np.nan

    max_tsteps = max_rt/dt

    x = np.zeros(drift.shape)
    x_1_2 = np.zeros(shape)
    ongoing_1_2 = np.array(np.ones(shape), dtype=bool)
    ended_1_2 = np.array(np.ones(shape), dtype=bool)
    x_1_3 = np.zeros(shape)
    ongoing_1_3 = np.array(np.ones(shape), dtype=bool)
    ended_1_3 = np.array(np.ones(shape), dtype=bool)

    x_2_1 = np.zeros(shape)
    ongoing_2_1 = np.array(np.ones(shape), dtype=bool)
    ended_2_1 = np.array(np.ones(shape), dtype=bool)
    x_2_3 = np.zeros(shape)
    ongoing_2_3 = np.array(np.ones(shape), dtype=bool)
    ended_2_3 = np.array(np.ones(shape), dtype=bool)

    x_3_1 = np.zeros(shape)
    ongoing_3_1 = np.array(np.ones(shape), dtype=bool)
    ended_3_1 = np.array(np.ones(shape), dtype=bool)
    x_3_2 = np.zeros(shape)
    ongoing_3_2 = np.array(np.ones(shape), dtype=bool)
    ended_3_2 = np.array(np.ones(shape), dtype=bool)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)
    ended = np.array(np.zeros(drift.shape), dtype=bool)

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        ong = np.logical_and(ongoing, ongoing_1_2)
        x_1_2[ong] += np.random.normal(drift[ong, 0]*dt,
                                       sigma[ong]*np.sqrt(dt),
                                       np.sum(ong))
        
        ong = np.logical_and(ongoing, ongoing_1_3)
        x_1_3[ong] += np.random.normal(drift[ong, 1]*dt,
                                       sigma[ong]*np.sqrt(dt),
                                       np.sum(ong))
        
        ong = np.logical_and(ongoing, ongoing_2_1)
        x_2_1[ong] += np.random.normal(drift[ong, 2]*dt,
                                       sigma[ong]*np.sqrt(dt),
                                       np.sum(ong))
        
        ong = np.logical_and(ongoing, ongoing_2_3)
        x_2_3[ong] += np.random.normal(drift[ong, 3]*dt,
                                       sigma[ong]*np.sqrt(dt),
                                       np.sum(ong))
        
        ong = np.logical_and(ongoing, ongoing_3_1)
        x_3_1[ong] += np.random.normal(drift[ong, 4]*dt,
                                       sigma[ong]*np.sqrt(dt),
                                       np.sum(ong))
        
        ong = np.logical_and(ongoing, ongoing_3_2)
        x_3_2[ong] += np.random.normal(drift[ong, 5]*dt,
                                       sigma[ong]*np.sqrt(dt),
                                       np.sum(ong))
        
        tstep += 1
        
        ended_1_2 = x_1_2 >= 1
        ended_1_3 = x_1_3 >= 1
        ended_2_1 = x_2_1 >= 1
        ended_2_3 = x_2_3 >= 1
        ended_3_1 = x_3_1 >= 1
        ended_3_2 = x_3_2 >= 1

        choice[np.logical_and(np.logical_and(ended_1_2, ended_1_3), ongoing)] = 1
        rt[np.logical_and(np.logical_and(ended_1_2, ended_1_3), ongoing)] = dt*tstep + ndt[np.logical_and(np.logical_and(ended_1_2, ended_1_3), ongoing)]
        ongoing_1_2[ended_1_2] = False
        ongoing_1_3[ended_1_3] = False
        ongoing[np.logical_and(ended_1_2, ended_1_3)] = False
        
        choice[np.logical_and(np.logical_and(ended_2_1, ended_2_3), ongoing)] = 2
        rt[np.logical_and(np.logical_and(ended_2_1, ended_2_3), ongoing)] = dt*tstep + ndt[np.logical_and(np.logical_and(ended_2_1, ended_2_3), ongoing)]
        ongoing_2_1[ended_2_1] = False
        ongoing_2_3[ended_2_3] = False
        ongoing[np.logical_and(ended_2_1, ended_2_3)] = False
        
        choice[np.logical_and(np.logical_and(ended_3_1, ended_3_2), ongoing)] = 3
        rt[np.logical_and(np.logical_and(ended_3_1, ended_3_2), ongoing)] = dt*tstep + ndt[np.logical_and(np.logical_and(ended_3_1, ended_3_2), ongoing)]
        ongoing_3_1[ended_3_1] = False
        ongoing_3_2[ended_3_2] = False
        ongoing[np.logical_and(ended_3_1, ended_3_2)] = False
        

    return rt, choice

def random_glam_2A(inc_drift, 
                   cor_drift, 
                  sigma, 
                  ndt, 
                  spvar=None,
                  starting_point_variability=False,
                  noise_constant=1, dt=0.001, max_rt=10):
    """ Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    Parameters
    ----------

    cor_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - correct option.

    inc_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - incorrect option.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold parameter.

    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time parameter, in seconds.

    spvar : numpy.ndarray, default None
        Shape is usually (n_samples, n_trials).
        Starting point variability paramter, in seconds.

    starting_point_variability : bool, defauld False
        Determines whether there starting point variability or not.

    noise_constant : float, default 1
        Scaling factor of the Racing Diffusion Model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the Racing Diffusion Model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """
    # Based on the Wiener diffusion process
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt


    if starting_point_variability:
        x_cor = np.random.uniform(0, spvar)
        x_inc = np.random.uniform(0, spvar)
    else:
        x_cor = np.zeros(shape)
        x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           sigma[ongoing] * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           sigma[ongoing] * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= 1)
        ended_incorrect = (x_inc >= 1)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc

def random_glam_nA(drift, sigma, ndt, noise_constant=1, dt=0.001, max_rt=10):
    shape = ndt.shape
    n_options = drift.shape[2]
    choice = np.empty(shape)*np.nan
    rt = np.empty(shape)*np.nan

    max_tsteps = max_rt/dt

    x = np.zeros(drift.shape)
    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)
    ended = np.array(np.zeros(drift.shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        for i in range(n_options):
            x[ongoing, i] += np.random.normal(drift[ongoing, i]*dt,
                                               sigma[ongoing]*np.sqrt(dt),
                                               np.sum(ongoing))
        tstep += 1
        # rt[ongoing] += dt

        for i in range(n_options):
            ended[:, :, i]= (x[:, :, i] >= 1)

        # store results and filter out ended trials
        for i in range(n_options-1, -1, -1):
            # print(i)
            ended_i = ended[:, :, i]
            for j in range(n_options):
                if i!=j:
                    ended_i = np.logical_and(ended_i, np.logical_not(ended[:, :, j]))
            if np.sum(ended_i) > 0:
                choice[np.logical_and(ended_i, ongoing)] = i + 1
                rt[np.logical_and(ended_i, ongoing)] = dt*tstep + ndt[np.logical_and(ended_i, ongoing)]
                ongoing[ended_i] = False
    return rt, choice

def random_ardm_2A(cor_drift, 
                  inc_drift, 
                  threshold_cor,
                  threshold_inc,
                  ndt,
                  data,
                  spvar=None,
                  starting_point_variability=False,
                  noise_constant=1, dt=0.001, max_rt=10):
    """ Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    Parameters
    ----------

    cor_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - correct option.

    inc_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - incorrect option.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold parameter.

    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time parameter, in seconds.

    spvar : numpy.ndarray, default None
        Shape is usually (n_samples, n_trials).
        Starting point variability paramter, in seconds.

    starting_point_variability : bool, defauld False
        Determines whether there starting point variability or not.

    noise_constant : float, default 1
        Scaling factor of the Racing Diffusion Model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the Racing Diffusion Model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """
    # Based on the Wiener diffusion process
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt


    if starting_point_variability:
        x_cor = np.random.uniform(0, spvar)
        x_inc = np.random.uniform(0, spvar)
    else:
        x_cor = np.zeros(shape)
        x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= threshold_cor)
        ended_incorrect = (x_inc >= threshold_inc)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc

def random_trdm_2A(cor_drift, 
                  inc_drift,
                  time_drift, 
                  threshold, 
                  ndt,
                  noise_constant=1, dt=0.001, max_rt=10):
    """ Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    Parameters
    ----------

    cor_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - correct option.

    inc_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - incorrect option.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold parameter.

    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time parameter, in seconds.

    starting_point_variability : bool, defauld False
        Determines whether there starting point variability or not.

    noise_constant : float, default 1
        Scaling factor of the Racing Diffusion Model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the Racing Diffusion Model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """
    # Based on the Wiener diffusion process
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt



    x_cor = np.zeros(shape)
    x_inc = np.zeros(shape)
    x_time = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))

        x_time[ongoing] += np.random.normal(time_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        
        ended_correct = (x_cor >= threshold)
        ended_incorrect = (x_inc >= threshold)
        ended_time = (x_time >= threshold)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False

        if np.sum(ended_time) > 0:
            acc[np.logical_and(ended_time, ongoing)] = np.random.randint(0, 2, shape)[np.logical_and(ended_time, ongoing)]
            rt[np.logical_and(ended_time, ongoing)] = dt * tstep + ndt[np.logical_and(ended_time, ongoing)]
            ongoing[ended_time] = False
    return rt, acc


def random_rdm_2A(cor_drift, 
                  inc_drift, 
                  threshold, 
                  ndt, 
                  spvar=None,
                  starting_point_variability=False,
                  noise_constant=1, dt=0.001, max_rt=10):
    """ Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    Parameters
    ----------

    cor_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - correct option.

    inc_drift : numpy.ndarray
        Drift-rate of the Racing Diffusion Model - incorrect option.

    threshold : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Threshold parameter.

    ndt : numpy.ndarray
        Shape is usually (n_samples, n_trials).
        Non decision time parameter, in seconds.

    spvar : numpy.ndarray, default None
        Shape is usually (n_samples, n_trials).
        Starting point variability paramter, in seconds.

    starting_point_variability : bool, defauld False
        Determines whether there starting point variability or not.

    noise_constant : float, default 1
        Scaling factor of the Racing Diffusion Model.
        If changed, drift and threshold would be scaled accordingly.
        Not to be changed in most applications.

    dt : float, default 0.001
        Controls the time resolution of the Racing Diffusion Model. Default is 1 msec.
        Lower values of dt make the function more precise but much slower.

    max_rt : float, default 10
        Controls the maximum rts that can be predicted.
        Making this higher might make the function a bit slower.

    Returns
    -------

    rt : numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated response times according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    acc: numpy.ndarray
        Shape is the same as the input parameters.
        Contains simulated accuracy according to the Racing Diffusion Model.
        Every element corresponds to the set of parameters given as input with the same shape.

    """
    # Based on the Wiener diffusion process
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt


    if starting_point_variability:
        x_cor = np.random.uniform(0, spvar)
        x_inc = np.random.uniform(0, spvar)
    else:
        x_cor = np.zeros(shape)
        x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= threshold)
        ended_incorrect = (x_inc >= threshold)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc


def simulate_rdm_2A(n_trials,
                    gen_cor_drift,
                    gen_inc_drift,
                    gen_threshold,
                    gen_ndt,
                    gen_spvar=None,
                    participant_label=1,
                    **kwargs):
    """Simulates behavior (rt and accuracy) according to the Racing Diffusion Model.

    This function is to simulate data for, for example, parameter recovery.

    Simulates data for one participant.

    Parameters
    ----------

    n_trials : int
        Number of trials to be simulated.

    gen_cor_drift : float
        Drift-rate of the Racing Diffusion Model - correct trials.

    gen_inc_drift : float
        Drift-rate of the Racing Diffusion Model - incorrect trials.

    gen_threshold : float
        Threshold of the Racing Diffusion Model.
        Should be positive.

    gen_ndt : float
        Non decision time of the Racing Diffusion Model, in seconds.
        Should be positive.

    participant_label : string or float, default 1
        What will appear in the participant column of the output data.

    Other Parameters
    ----------------

    **kwargs
        Additional arguments to rlssm.random.random_rdm_2A().

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, with n_trials rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters
        (both for each trial and across-trials when there is across-trial variability).
    """
    # return a pandas dataframe with the following columns:
    # index: participant + trial, cor_drift, inc_drift, threshold, ndt, rt, accuracy
    data = pd.DataFrame({'participant': np.repeat(participant_label, n_trials)})

    data['cor_drift'] = gen_cor_drift
    data['inc_drift'] = gen_inc_drift

    data['threshold'] = gen_threshold
    data['ndt'] = gen_ndt
    if gen_spvar is None:
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'],
                                data['threshold'],
                                data['ndt'],
                                **kwargs)
    else:
        data['spvar'] = gen_spvar
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'],
                                data['threshold'],
                                data['ndt'],
                                spvar=data['spvar'],
                                starting_point_variability=True,
                                **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.arange(1, n_trials + 1)

    data = data.set_index(['participant', 'trial'])

    return data


def simulate_hier_rdm(n_trials, n_participants,
                      gen_mu_drift_cor, gen_sd_drift_cor,
                      gen_mu_drift_inc, gen_sd_drift_inc,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      gen_mu_spvar=None, gen_sd_spvar=None,
                      **kwargs):
    """Simulates behavior (rt and accuracy) according to the Racing Difussion Model.

    Parameters
    ----------

    n_trials : int
        Number of trials to simulate.

    n_participants : int
        Number of participants to simulate.

    gen_mu_drift_cor : float
        Group-mean of the drift-rate of the RDM for the correct responses.

    gen_sd_drift_cor : float
        Group-standard deviation of the drift-rate of the RDM for the correct responses.

    gen_mu_drift_inc : float
        Group-mean of the drift-rate of the RDM for the incorrect responses.

    gen_sd_drift_inc : float
        Group-standard deviation of the drift-rate of the RDM for the incorrect responses.

    gen_mu_threshold : float
        Group-mean of the threshold of the RDM.

    gen_sd_threshold : float
        Group-standard deviation of the threshold of the RDM.

    gen_mu_ndt : float
        Group-mean of the non-decision time of the RDM.

    gen_sd_ndt : float
        Group-standard deviation of the non-decision time of the RDM.

    gen_mu_spvar : float
        Group-mean of the starting point variability parameter.

    gen_sd_spvar : float
        Group-standard deviation of starting point variability parameter.

    Other Parameters
    ----------------

    **kwargs : dict
        Keyword arguments to be passed to `random_rdm_2A`.

    Returns
    -------

    data : pandas.DataFrame
        `pandas.DataFrame`, with n_trials*n_participants rows.
        Columns contain simulated response times and accuracy ["rt", "accuracy"],
        as well as the generating parameters (at the participant level).
    """
    data = pd.DataFrame([])

    cor_drift_sbj = np.random.normal(gen_mu_drift_cor, gen_sd_drift_cor, n_participants)
    inc_drift_sbj = np.random.normal(gen_mu_drift_inc, gen_sd_drift_inc, n_participants)

    threshold_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_threshold, gen_sd_threshold, n_participants)))
    ndt_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_ndt, gen_sd_ndt, n_participants)))

    
    data['participant'] = np.repeat(np.arange(n_participants) + 1, n_trials)
    data['cor_drift'] = np.repeat(cor_drift_sbj, n_trials)
    data['inc_drift'] = np.repeat(inc_drift_sbj, n_trials)
    data['threshold'] = np.repeat(threshold_sbj, n_trials)
    data['ndt'] = np.repeat(ndt_sbj, n_trials)
    
    if gen_mu_spvar is None or gen_sd_spvar is None:
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'],
                                data['threshold'],
                                data['ndt'],
                                **kwargs)
    else:
        spvar_sbj = np.log(1 + np.exp(np.random.normal(gen_mu_spvar, gen_sd_spvar, n_participants)))
        data['spvar'] = np.repeat(spvar_sbj, n_trials)
        rt, acc = random_rdm_2A(data['cor_drift'],
                                data['inc_drift'],
                                data['threshold'],
                                data['ndt'],
                                spvar=data['spvar'],
                                starting_point_variability=True,
                                **kwargs)

    data['rt'] = rt
    data['accuracy'] = acc
    data['trial'] = np.tile(np.arange(1, n_trials + 1), n_participants)

    data = data.set_index(['participant', 'trial'])

    return data
