from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from assm.fit.fits import FittedModel, ModelResults
from assm.plot import plotting
from assm.random.random_RDM import random_rdm_2A, random_gard_2A, random_trdm_2A, random_glam_2A, random_ardm_2A
from assm.utility.utils import list_individual_variables


class RDMFittedModel_2A(FittedModel):
    def __init__(self,
                 stan_model,
                 data,
                 hierarchical_levels,
                 model_label,
                 family,
                 n_parameters_individual,
                 n_parameters_trial,
                 print_diagnostics,
                 priors,
                 starting_point_variability):
        self.family = family
        super().__init__(stan_model,
                         data,
                         hierarchical_levels,
                         model_label,
                         family,
                         n_parameters_individual,
                         n_parameters_trial,
                         print_diagnostics,
                         priors)
        self.starting_point_variability = starting_point_variability

    def extract_results(self, include_rhat, include_waic, pointwise_waic, include_last_values):
        if include_rhat:
            rhat = self.get_rhat()
        else:
            rhat = None

        if include_waic:
            waic = self.calculate_waic(pointwise_waic)
        else:
            waic = None

        if include_last_values:
            last_values = self.get_last_values()
        else:
            last_values = None

        # main parameters
        if self.parameters_info['hierarchical_levels'] == 2:
            main_parameters = self.parameters_info['group_parameters_names_transf']

            main_parameters = np.append(main_parameters,
                                        [p + '_sbj' for p in self.parameters_info['individual_parameters_names']])

            # for p in self.parameters_info['individual_parameters_names']:
            #     main_parameters = np.append(main_parameters, list_individual_variables(p, self.data_info['L']))
        else:
            main_parameters = self.parameters_info['parameters_names_transf']

        samples = self.stan_model.draws_pd(vars=main_parameters)
        if self.model_label.find('TBGLAM_2A') != -1:
            trial_samples = {'drift_cor_t': np.asarray(self.stan_model.draws_pd(vars=['drift_cor_t'])),
                            'drift_inc_t': np.asarray(self.stan_model.draws_pd(vars=['drift_inc_t'])),
                            'drift_time_t': np.asarray(self.stan_model.draws_pd(vars=['drift_time_t'])),
                            'threshold_t': np.asarray(self.stan_model.draws_pd(vars=['threshold_t']))}

        elif self.model_label.find('GLAM_2A') != -1 and self.model_label.find('TBGLAM_2A') == -1:
            trial_samples = {'drift_left_t': np.asarray(self.stan_model.draws_pd(vars=['drift_left_t'])),
                             'drift_right_t': np.asarray(self.stan_model.draws_pd(vars=['drift_right_t'])),
                             'sigma_t': np.asarray(self.stan_model.draws_pd(vars=['sigma_t'])),
                             'ndt_t': np.asarray(self.stan_model.draws_pd(vars=['ndt_t']))}
            
        elif self.model_label.find('GARD_2A') != -1:
            trial_samples = {'drift_left_t': np.asarray(self.stan_model.draws_pd(vars=['drift_left_t'])),
                             'drift_right_t': np.asarray(self.stan_model.draws_pd(vars=['drift_right_t'])),
                             'sigma_t': np.asarray(self.stan_model.draws_pd(vars=['sigma_t'])),
                             'ndt_t': np.asarray(self.stan_model.draws_pd(vars=['ndt_t']))}

        elif self.model_label.find('ARDM_2A') != -1:
            trial_samples = {'drift_cor_t': np.asarray(self.stan_model.draws_pd(vars=['drift_cor_t'])),
                            'drift_inc_t': np.asarray(self.stan_model.draws_pd(vars=['drift_inc_t'])),
                            'ndt_t':np.asarray(self.stan_model.draws_pd(vars=['ndt_t'])),
                            'threshold_cor_t': np.asarray(self.stan_model.draws_pd(vars=['threshold_cor_t'])),
                            'threshold_inc_t': np.asarray(self.stan_model.draws_pd(vars=['threshold_inc_t']))}

        elif self.starting_point_variability:
            trial_samples = {'drift_cor_t': np.asarray(self.stan_model.draws_pd(vars=['drift_cor_t'])),
                            'drift_inc_t': np.asarray(self.stan_model.draws_pd(vars=['drift_inc_t'])),
                            'threshold_t': np.asarray(self.stan_model.draws_pd(vars=['threshold_t'])),
                            'ndt_t': np.asarray(self.stan_model.draws_pd(vars=['ndt_t'])),
                            'sp_trial_var_t':np.asarray(self.stan_model.draws_pd(vars=['sp_trial_var_t']))}
        else:
            trial_samples = {'drift_cor_t': np.asarray(self.stan_model.draws_pd(vars=['drift_cor_t'])),
                            'drift_inc_t': np.asarray(self.stan_model.draws_pd(vars=['drift_inc_t'])),
                            'threshold_t': np.asarray(self.stan_model.draws_pd(vars=['threshold_t']))}
                            # 'ndt_t': np.asarray(self.stan_model.draws_pd(vars=['ndt_t']))}

        res = RDMModelResults_2A(self.model_label,
                                 self.data_info,
                                 self.parameters_info,
                                 self.priors,
                                 rhat,
                                 waic,
                                 last_values,
                                 samples,
                                 trial_samples,
                                 self.family,
                                 self.starting_point_variability)
        return res


class RDMModelResults_2A(ModelResults):
    def __init__(self,
                 model_label,
                 data_info,
                 parameters_info,
                 priors,
                 rhat,
                 waic,
                 last_values,
                 samples,
                 trial_samples,
                 family,
                 starting_point_variability):
        self.family = family
        super().__init__(model_label,
                         data_info,
                         parameters_info,
                         priors,
                         rhat,
                         waic,
                         last_values,
                         samples,
                         trial_samples)
        self.starting_point_variability = starting_point_variability
        self.pp_rt = None
        self.pp_choice = None
        self.pp_df = None
        self.is_pp_calculated = False

    def get_posterior_predictives(self, n_posterior_predictives=500, **kwargs):
        if n_posterior_predictives > self.parameters_info['n_posterior_samples']:
            warnings.warn("Cannot have more posterior predictive samples than posterior samples. " \
                          "Will continue with n_posterior_predictives=%s" % self.parameters_info['n_posterior_samples'],
                          UserWarning,
                          stacklevel=2)
            n_posterior_predictives = self.parameters_info['n_posterior_samples']

        drift_left_t = self.trial_samples['drift_left_t'][:n_posterior_predictives, :]
        drift_right_t = self.trial_samples['drift_right_t'][:n_posterior_predictives, :]

        # ndt_t = np.zeros(drift_left_t.shape) # self.trial_samples['ndt_t'][:n_posterior_predictives, :]
        ndt_t = self.trial_samples['ndt_t'][:n_posterior_predictives, :]
        if self.model_label.find('TBGLAM_2A') != -1:
            threshold_t = self.trial_samples['threshold_t'][:n_posterior_predictives, :]
            drift_time_t = self.trial_samples['drift_time_t'][:n_posterior_predictives, :]
            pp_rt, pp_acc = random_trdm_2A(drift_cor_t, drift_inc_t, drift_time_t, threshold_t, ndt_t, **kwargs)

        elif self.model_label.find('GLAM_2A') != -1 and self.model_label.find('TBGLAM_2A') == -1:
            sigma_t = self.trial_samples['sigma_t'][:n_posterior_predictives, :]
            pp_rt, pp_acc = random_glam_2A(drift_left_t, drift_right_t, sigma_t, ndt_t, **kwargs)

        elif self.model_label.find('GARD_2A') != -1:
            sigma_t = self.trial_samples['sigma_t'][:n_posterior_predictives, :]
            pp_rt, pp_acc = random_gard_2A(drift_left_t, drift_right_t, sigma_t, ndt_t, **kwargs)
        
        elif self.model_label.find('ARDM_2A') != -1:
            ndt_t = self.trial_samples['ndt_t'][:n_posterior_predictives, :]
            threshold_cor_t = self.trial_samples['threshold_cor_t'][:n_posterior_predictives, :]
            threshold_inc_t = self.trial_samples['threshold_inc_t'][:n_posterior_predictives, :]
            pp_rt, pp_acc = random_ardm_2A(drift_cor_t, drift_inc_t, threshold_cor_t, threshold_inc_t, ndt_t, self.data_info['data'], **kwargs)

        elif self.starting_point_variability:
            spvar_t = self.trial_samples['sp_trial_var_t'][:n_posterior_predictives, :]
            pp_rt, pp_acc = random_rdm_2A(drift_cor_t, drift_inc_t, threshold_t, ndt_t, spvar_t, True, **kwargs)
        else:
            threshold_t = self.trial_samples['threshold_t'][:n_posterior_predictives, :]
            pp_rt, pp_acc = random_rdm_2A(drift_cor_t, drift_inc_t, threshold_t, ndt_t, **kwargs)

        return pp_rt, pp_acc

    def get_posterior_predictives_df(self, n_posterior_predictives=500, **kwargs):
        """Calculates posterior predictives of choices and response times.

        Parameters
        ----------

        n_posterior_predictives : int, default 500
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the posterior predictive function.

        Returns
        -------

        out : DataFrame
            Data frame of shape (n_samples, n_trials*2).
            Response times and accuracy are provided as hierarchical column indeces.

        """
        if not self.is_pp_calculated:
            pp_rt, pp_acc = self.get_posterior_predictives(n_posterior_predictives, **kwargs)

            tmp1 = pd.DataFrame(pp_rt,
                                index=pd.Index(np.arange(1, len(pp_rt) + 1), name='sample'),
                                columns=pd.MultiIndex.from_product((['rt'],
                                                                    np.arange(pp_rt.shape[1]) + 1),
                                                                names=['variable', 'trial']))
            tmp2 = pd.DataFrame(pp_acc,
                                index=pd.Index(np.arange(1, len(pp_acc) + 1), name='sample'),
                                columns=pd.MultiIndex.from_product((['accuracy'],
                                                                    np.arange(pp_acc.shape[1]) + 1),
                                                                names=['variable', 'trial']))
            out = pd.concat((tmp1, tmp2), axis=1)
            self.pp_df = out
            self.is_pp_calculated = True
        else:
            out = self.pp_df
        return out

    def get_posterior_predictives_summary(self,
                                          n_posterior_predictives=500,
                                          quantiles=None,
                                          **kwargs):
        """Calculates summary of posterior predictives of choices and response times. The mean proportion of choices
        (in this case coded as accuracy) is calculated for each posterior sample across all trials. Response times
        are summarized using mean, skewness, and quantiles.

        Parameters
        ----------

        n_posterior_predictives : int, default 500
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        quantiles : list of floats
             Quantiles to summarize response times distributions
             (separately for correct/incorrect) with.
             Default to [.1, .3, .5, .7, .9].

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `self.get_posterior_predictives_df`.

        noise_constant : float
            Scaling factor of the diffusion decision model.
            If changed, drift and threshold would be scaled accordingly.
            Not to be changed in most applications.

        rt_max : float
            Controls the maximum rts that can be predicted.
            Making this higher might make the function a bit slower.

        dt : float
            Controls the time resolution of the diffusion decision model. Default is 1 msec.
            Lower values of dt make the function more precise but much slower.

        Returns
        -------

        out : DataFrame
            Pandas DataFrame, where every row corresponds to a posterior sample.
            The columns contains the mean accuracy for each posterior sample,
            as well as mean response times, response times skewness and response times quantiles.

        """
        if quantiles is None:
            quantiles = [.1, .3, .5, .7, .9]

        pp = self.get_posterior_predictives_df(n_posterior_predictives=n_posterior_predictives, **kwargs)

        tmp = pd.DataFrame({'mean_accuracy': pp['accuracy'].mean(axis=1),
                            'mean_rt': pp['rt'].mean(axis=1),
                            'skewness': pp['rt'].skew(axis=1, skipna=True)})

        pp_rt_inc = pp['rt'][pp['accuracy'] == 0]
        pp_rt_cor = pp['rt'][pp['accuracy'] == 1]

        q_inc = pp_rt_inc.quantile(q=quantiles, axis=1).T
        q_cor = pp_rt_cor.quantile(q=quantiles, axis=1).T

        q_inc.columns = ['quant_{}_rt_incorrect'.format(int(c * 100)) for c in q_inc.columns]
        q_cor.columns = ['quant_{}_rt_correct'.format(int(c * 100)) for c in q_cor.columns]

        out = pd.concat([tmp, q_inc, q_cor], axis=1)

        return out

    def plot_mean_posterior_predictives(self,
                                        n_posterior_predictives,
                                        figsize=(20, 8),
                                        post_pred_kws=None,
                                        **kwargs):
        """Plots the mean posterior predictives of choices and response times.

        The mean proportion of choices (in this case coded as accuracy) is calculated
        for each posterior sample across all trials,
        and then it's plotted as a distribution.
        The mean accuracy in the data is plotted as vertical line.
        This allows to compare the real mean with the BCI or HDI of the predictions.
        The same is done for response times, and are plotted one next to each other.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        figsize : tuple, default (20, 8)
            figure size of the matplotlib figure

        post_pred_kws : dict, default None
            Additional parameters to get_posterior_predictives_summary.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `plotting.plot_mean_prediction`.

        Returns
        -------

        fig : matplotlib.figure.Figure

        """
        if post_pred_kws is None:
            post_pred_kws = {}

        pp_df = self.get_posterior_predictives_summary(n_posterior_predictives, **post_pred_kws)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        plotting.plot_mean_prediction(pp_df,
                                      self.data_info['data'],
                                      y_data='accuracy',
                                      y_predictions='mean_accuracy',
                                      ax=axes[0],
                                      **kwargs)

        plotting.plot_mean_prediction(pp_df,
                                      self.data_info['data'],
                                      y_data='rt',
                                      y_predictions='mean_rt',
                                      ax=axes[1],
                                      **kwargs)

        axes[0].set_xlabel('Mean accuracy')
        axes[1].set_xlabel('Mean RTs')
        axes[0].set_ylabel('Density')
        sns.despine()
        return fig

    def plot_quantiles_posterior_predictives(self,
                                             n_posterior_predictives,
                                             quantiles=None,
                                             figsize=(20, 8),
                                             post_pred_kws=None,
                                             **kwargs):
        """Plots the quantiles of the posterior predictives of response times,
        separately for correct/incorrect responses.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        quantiles : list of floats, default None
             Quantiles to summarize response times distributions (separately for correct/incorrect) with.

        figsize : tuple, default (20, 8)
            figure size of the matplotlib figure

        post_pred_kws : dict, default None
            Additional parameters to `get_posterior_predictives_summary`.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `plotting.plot_quantiles_prediction`.

        show_data : bool
            Whether to show the quantiles of the data. Set to False to not show it.

        show_intervals : either "HDI", "BCI", or None
            HDI is better when the distribution is not symmetrical.
            If None, then no intervals are shown.

        alpha_intervals : float
            Alpha level for the intervals.
            Default is 5 percent which gives 95 percent BCIs and HDIs.

        kind : either 'lines' or 'shades'
            Two different styles to plot quantile distributions.

        color : matplotlib color
            Color for both the data and intervals.

        scatter_kws : dictionary
            Additional plotting parameters to change how the data points are shown.

        intervals_kws : dictionary
            Additional plotting parameters to change how the quantile distributions are shown.

        post_pred_kws : dictionary
            Additional parameters to get_posterior_predictives_summary.

        Returns
        -------

        fig : matplotlib.figure.Figure

        """

        if post_pred_kws is None:
            post_pred_kws = {}

        pp_summary = self.get_posterior_predictives_summary(
            n_posterior_predictives=n_posterior_predictives,
            quantiles=quantiles,
            **post_pred_kws)

        fig = plotting.plot_quantiles_prediction(pp_summary,
                                                 self.data_info['data'],
                                                 'rdm',
                                                 quantiles=quantiles,
                                                 figsize=figsize,
                                                 **kwargs)
        return fig

    def get_grouped_posterior_predictives_summary(self,
                                                  grouping_vars,
                                                  n_posterior_predictives=500,
                                                  quantiles=None,
                                                  **kwargs):
        """Calculates summary of posterior predictives of choices and response times,
        separately for a list of grouping variables.

        The mean proportion of choices (in this case coded as accuracy) is calculated
        for each posterior sample across all trials
        in all conditions combination.
        Response times are summarized using mean, skewness, and quantiles.

        For example, if grouping_vars=['reward', 'difficulty'],
        posterior predictives will be collapsed
        for all combinations of levels of the reward and difficulty variables.

        Parameters
        ----------

        grouping_vars :  list of strings
             They should be existing grouping variables in the data.

        n_posterior_predictives : int, default 500
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        quantiles : list of floats, default None
             Quantiles to summarize response times distributions
             (separately for correct/incorrect) with.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `self.get_posterior_predictives_df`.

        noise_constant : float
            Scaling factor of the diffusion decision model.
            If changed, drift and threshold would be scaled accordingly.
            Not to be changed in most applications.

        rt_max : float
            Controls the maximum rts that can be predicted.
            Making this higher might make the function a bit slower.

        dt : float
            Controls the time resolution of the diffusion decision model. Default is 1 msec.
            Lower values of dt make the function more precise but much slower.

        Returns
        -------

        out : DataFrame
             Pandas DataFrame.
             The columns contains the mean accuracy for each posterior sample,
             as well as mean response times, response times skewness and response times quantiles.
             The row index is a pandas.MultIndex, with the grouping variables as higher level
             and number of samples as lower level.

        """
        if quantiles is None:
            quantiles = [.1, .3, .5, .7, .9]

        data_copy = self.data_info['data'].copy()
        data_copy['trial'] = np.arange(1, self.data_info['N'] + 1)
        data_copy.set_index('trial', inplace=True)

        pp = self.get_posterior_predictives_df(n_posterior_predictives=n_posterior_predictives,
                                               **kwargs)

        tmp = pp.copy().T.reset_index().set_index('trial')

        tmp = pd.merge(tmp,
                       data_copy[grouping_vars],
                       left_index=True,
                       right_index=True).reset_index()
        tmp_rt = tmp[tmp.variable == 'rt'].drop('variable', axis=1)
        tmp_accuracy = tmp[tmp.variable == 'accuracy'].drop('variable', axis=1)

        out = pd.concat([tmp_accuracy.groupby(grouping_vars).mean().drop('trial',
                                                                         axis=1).stack().to_frame('mean_accuracy'),
                         tmp_rt.groupby(grouping_vars).mean().drop('trial',
                                                                   axis=1).stack().to_frame('mean_rt'),
                         tmp_rt.groupby(grouping_vars).skew().drop('trial',
                                                                   axis=1).stack().to_frame('skewness')],
                        axis=1)

        tmp_accuracy.set_index(list(np.append(grouping_vars, 'trial')), inplace=True)
        tmp_rt.set_index(list(np.append(grouping_vars, 'trial')), inplace=True)

        pp_rt_low = tmp_rt[tmp_accuracy == 0]  # lower boundary (usually incorrect)
        pp_rt_up = tmp_rt[tmp_accuracy == 1]  # upper boundary (usually correct)

        for q in quantiles:
            new_col = 'quant_{}_rt_incorrect'.format(int(q * 100))

            out[new_col] = pp_rt_low.reset_index().groupby(grouping_vars).quantile(q).drop('trial',
                                                                                           axis=1).stack().to_frame(
                'quant')

            new_col = 'quant_{}_rt_correct'.format(int(q * 100))

            out[new_col] = pp_rt_up.reset_index().groupby(grouping_vars).quantile(q).drop('trial',
                                                                                          axis=1).stack().to_frame(
                'quant')

        out.index.rename(np.append(grouping_vars, 'sample'), inplace=True)

        return out

    def plot_mean_grouped_posterior_predictives(self,
                                                grouping_vars,
                                                n_posterior_predictives,
                                                figsize=(20, 8),
                                                n_rows=1,
                                                row=1,
                                                post_pred_kws=None,
                                                fig=None,
                                                axes=None,
                                                show_rt=True,
                                                show_ch=True,
                                                **kwargs):
        """Plots the mean posterior predictives of choices and response times,
        separately for either 1 or 2 grouping variables.

        The first grouping variable will be plotted on the x-axis.
        The second grouping variable, if provided, will be showed
        with a different color per variable level.

        Parameters
        ----------

        grouping_vars :  list of strings
             They should be existing grouping variables in the data.
             The list should be of lenght 1 or 2.

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        figsize : tuple, default (20, 8)
            figure size of the matplotlib figure

        post_pred_kws : dict, default None
            Additional parameters to get_posterior_predictives_summary.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `plotting.plot_grouped_mean_prediction`.

        Returns
        -------

        fig : matplotlib.figure.Figure

        """

        if np.sum(len(grouping_vars) == np.array([1, 2])) < 1:
            raise ValueError("must be a list of either 1 or values")

        if post_pred_kws is None:
            post_pred_kws = {}

        pp = self.get_grouped_posterior_predictives_summary(grouping_vars,
                                                            n_posterior_predictives,
                                                            **post_pred_kws)

        if len(grouping_vars) == 1:
            if axes is None:
                fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

            if axes.ndim == 2:
                ax = axes[row-1, :]

            if show_ch:
                plotting.plot_grouped_mean_prediction(x=grouping_vars[0],
                                                    y_data='Right',
                                                    y_predictions='mean_accuracy',
                                                    predictions=pp,
                                                    data=self.data_info['data'],
                                                    ax=ax[0],
                                                    **kwargs)

            if show_rt:
                plotting.plot_grouped_mean_prediction(x=grouping_vars[0],
                                                    y_data='rt',
                                                    y_predictions='mean_rt',
                                                    predictions=pp,
                                                    data=self.data_info['data'],
                                                    ax=ax[1],
                                                    **kwargs)
        else:

            # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
            # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            if axes is None:
                fig, axes = plt.subplots(1, 2, figsize=figsize)

            plotting.plot_grouped_mean_prediction(x=grouping_vars[0],
                                                  y_data='Right',
                                                  y_predictions='mean_accuracy',
                                                  predictions=pp,
                                                  data=self.data_info['data'],
                                                  hue=grouping_vars[1],
                                                  ax=ax[0],
                                                  **kwargs)

            plotting.plot_grouped_mean_prediction(x=grouping_vars[0],
                                                  y_data='rt',
                                                  y_predictions='mean_rt',
                                                  predictions=pp,
                                                  data=self.data_info['data'],
                                                  hue=grouping_vars[1],
                                                  ax=ax[1],
                                                  **kwargs)
            ax[0].get_legend().remove()
            ax[1].legend(bbox_to_anchor=(1, 1))

        ax[0].set_ylabel('P(Choose Right)')
        ax[0].xaxis.label.set_fontsize(30)
        ax[0].yaxis.label.set_fontsize(30)

        for item in ax[0].get_xticklabels() + ax[0].get_yticklabels():
            item.set_fontsize(20)
        
        ax[1].set_ylabel('Mean RTs')
        ax[1].xaxis.label.set_fontsize(30)
        ax[1].yaxis.label.set_fontsize(30)

        for item in ax[1].get_xticklabels() + ax[1].get_yticklabels():
            item.set_fontsize(20)

        sns.despine()
        return fig, axes

    def plot_quantiles_grouped_posterior_predictives(self,
                                                     n_posterior_predictives,
                                                     grouping_var,
                                                     quantiles=None,
                                                     figsize=(20, 8),
                                                     post_pred_kws=None,
                                                     **kwargs):
        """Plots the quantiles of the posterior predictives of response times,
        separately for correct/incorrect responses, and 1 grouping variable.

        Parameters
        ----------

        n_posterior_predictives : int
             Number of posterior samples to use for posterior predictives calculation.
             If n_posterior_predictives is bigger than the posterior samples,
             then calculation will continue with the total number of posterior samples.

        grouping_var :  string
             Should be an existing grouping variable in the data.

        quantiles : list of floats, default None
             Quantiles to summarize response times distributions
             (separately for correct/incorrect) with.

        figsize : tuple, default (20, 8)
            figure size of the matplotlib figure

        post_pred_kws : dict, default None
            Additional parameters to get_posterior_predictives_summary.

        Other Parameters
        ----------------

        **kwargs : dict
            Keyword arguments to be passed to the `plotting.plot_grouped_quantiles_prediction`.

        Returns
        -------

        fig : matplotlib.figure.Figure

        """

        if post_pred_kws is None:
            post_pred_kws = {}

        pp_summary = self.get_grouped_posterior_predictives_summary(
            n_posterior_predictives=n_posterior_predictives,
            grouping_vars=[grouping_var],
            quantiles=quantiles,
            **post_pred_kws)

        fig = plotting.plot_grouped_quantiles_prediction(pp_summary,
                                                         self.data_info['data'],
                                                         'rdm',
                                                         quantiles=quantiles,
                                                         grouping_var=grouping_var,
                                                         figsize=figsize,
                                                         **kwargs)
        return fig
