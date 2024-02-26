from __future__ import absolute_import, division, print_function
import pandas as pd

from assm.model.models import Model
from assm.fit.fits_DDM import DDMFittedModel


class RLDDModel(Model):

    def __init__(self,
                 hierarchical_levels,
                 nonlinear_mapping=False,
                 separate_learning_rates=False,
                 threshold_modulation=False,
                 mapping_type=None,
                 modulation_type=None):
        
        super().__init__(hierarchical_levels, "aDDM")

        # Define the model parameters
        self.nonlinear_mapping = nonlinear_mapping
        self.separate_learning_rates = separate_learning_rates
        self.threshold_modulation = threshold_modulation
        self.mapping_type = mapping_type

        self.n_parameters_individual = 4
        self.n_parameters_trial = 0

        # Define default priors
        if self.hierarchical_levels == 1:
            self.priors = dict(
                alpha_priors={'mu': 0, 'sd': 1},
                alpha_pos_priors={'mu': 0, 'sd': 1},
                alpha_neg_priors={'mu': 0, 'sd': 1},
                drift_scaling_priors={'mu': 1, 'sd': 50},
                drift_asymptote_priors={'mu': 1, 'sd': 50},
                threshold_priors={'mu': 1, 'sd': 5},
                threshold_modulation_priors={'mu': 0, 'sd': 10},
                ndt_priors={'mu': 1, 'sd': 1},
                drift_power_priors={'mu': 0, 'sd': 0.5},
                threshold_power_priors={'mu': 0, 'sd': 0.5}
            )
        else:
            self.priors = dict(
                alpha_priors={'mu_mu': 0, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': .1},
                alpha_pos_priors={'mu_mu': 0, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': .1},
                alpha_neg_priors={'mu_mu': 0, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': .1},
                drift_scaling_priors={'mu_mu': 1, 'sd_mu': 30, 'mu_sd': 0, 'sd_sd': 30},
                drift_asymptote_priors={'mu_mu': 1, 'sd_mu': 30, 'mu_sd': 0, 'sd_sd': 30},
                threshold_priors={'mu_mu': 1, 'sd_mu': 3, 'mu_sd': 0, 'sd_sd': 3},
                threshold_modulation_priors={'mu_mu': 0, 'sd_mu': 10, 'mu_sd': 0, 'sd_sd': 10},
                ndt_priors={'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1},
                drift_power_priors={'mu_mu': 0, 'sd_mu': 0.5, 'mu_sd': 0.5, 'sd_sd': 0.5},
                threshold_power_priors={'mu_mu': 0, 'sd_mu': 0.5, 'mu_sd': 0.5, 'sd_sd': 0.5}
            )

        if self.nonlinear_mapping:
            if mapping_type == 'sigmoid':
                self.model_label += '_nonlin'
                self.priors.pop('drift_power_priors', None)
            elif mapping_type == 'power':
                self.model_label += '_driftpow'
                self.priors.pop('drift_asymptote_priors', None)
            self.n_parameters_individual += 1
        else:
            self.priors.pop('drift_asymptote_priors', None)
            self.priors.pop('drift_power_priors', None)

        if self.separate_learning_rates:
            self.model_label += '_2lr'
            self.n_parameters_individual += 1
            self.priors.pop('alpha_priors', None)
        else:
            self.priors.pop('alpha_pos_priors', None)
            self.priors.pop('alpha_neg_priors', None)

        if self.threshold_modulation:
            if modulation_type == 'exponential':
                self.model_label += '_thrmod'
                self.priors.pop('threshold_power_priors', None)
            elif modulation_type == 'power':
                self.model_label += '_thrpow'
                self.priors.pop('threshold_modulation_priors', None)
            self.n_parameters_individual += 1
        else:
            self.priors.pop('threshold_modulation_priors', None)
            self.priors.pop('threshold_power_priors', None)

        # Set the stan model path
        self._set_model_path()

        # Finally, compile the model
        self._compile_stan_model()

    def fit(self,
            data,
            K,
            initial_value_learning,
            alpha_priors=None,
            drift_scaling_priors=None,
            threshold_priors=None,
            ndt_priors=None,
            drift_asymptote_priors=None,
            threshold_modulation_priors=None,
            alpha_pos_priors=None,
            alpha_neg_priors=None,
            include_rhat=True,
            include_waic=True,
            pointwise_waic=False,
            include_last_values=True,
            print_diagnostics=True,
            **kwargs):
       
        data.reset_index(inplace=True)
        N = data.shape[0]  # n observations

        # transform data variables
        data['accuracy_neg'] = -1
        data.loc[data.accuracy == 1, 'accuracy_neg'] = 1

        # change default priors:
        if alpha_priors is not None:
            self.priors['alpha_priors'] = alpha_priors
        if alpha_pos_priors is not None:
            self.priors['alpha_pos_priors'] = alpha_pos_priors
        if alpha_neg_priors is not None:
            self.priors['alpha_neg_priors'] = alpha_neg_priors
        if drift_scaling_priors is not None:
            self.priors['drift_scaling_priors'] = drift_scaling_priors
        if drift_asymptote_priors is not None:
            self.priors['drift_asymptote_priors'] = drift_asymptote_priors
        if threshold_priors is not None:
            self.priors['threshold_priors'] = threshold_priors
        if threshold_modulation_priors is not None:
            self.priors['threshold_modulation_priors'] = threshold_modulation_priors
        if ndt_priors is not None:
            self.priors['ndt_priors'] = ndt_priors

        data_dict = {'N': N,
                     'K': K,
                     'trial_block': data['trial_block'].values.astype(int),
                     'f_cor': data['f_cor'].values,
                     'f_inc': data['f_inc'].values,
                     'cor_option': data['cor_option'].values.astype(int),
                     'inc_option': data['inc_option'].values.astype(int),
                     'block_label': data['block_label'].values.astype(int),
                     'rt': data['rt'].values,
                     'accuracy': data['accuracy_neg'].values.astype(int),
                     'feedback_type': data['feedback_type'].values.astype(int),
                     'initial_value': initial_value_learning,
                     'starting_point': .5}

        if self.hierarchical_levels == 2:
            keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
            L = len(pd.unique(data.participant))  # n subjects (levels)
            data_dict.update({'L': L,
                              'participant': data['participant'].values.astype(int)})
        else:
            keys_priors = ["mu", "sd"]

        # Add priors:
        print("Fitting the model using the priors:")
        for par in self.priors.keys():
            data_dict.update({par: [self.priors[par][key] for key in keys_priors]})
            print(par, self.priors[par])

        # start sampling...
        fitted_model = self.compiled_model.sample(data_dict, **kwargs)

        fitted_model = DDMFittedModel(fitted_model,
                                      data,
                                      self.hierarchical_levels,
                                      self.model_label,
                                      self.family,
                                      self.n_parameters_individual,
                                      self.n_parameters_trial,
                                      print_diagnostics,
                                      self.priors,
                                      False,
                                      False,
                                      False,
                                      False,
                                      False,
                                      False)
        res = fitted_model.extract_results(include_rhat,
                                           include_waic,
                                           pointwise_waic,
                                           include_last_values)

        return res
