from __future__ import absolute_import, division, print_function
import pandas as pd

from assm.fit.fits_RDM import RDMFittedModel_2A
from assm.fit.fits_nRDM import RDMFittedModel_nA
from assm.model.models import Model


class GARDModel_2A(Model):
    def __init__(self, hierarchical_levels,
                 additive=True,
                 multiplicative=True):
        super().__init__(hierarchical_levels, "GARD_2A")

        self.additive = additive
        self.multiplicative = multiplicative

        # Define the model parameters
        self.n_parameters_individual = 5  # sigma, v, wd, ws, ndt
        self.n_parameters_trial = 0

        # Define default priors
        if self.hierarchical_levels == 1:
            self.priors = dict(sigma_priors={'mu': 0, 'sd': 3},
                               v_priors={'mu': 9, 'sd': 3},
                               wd_priors={'mu': 1, 'sd': 2},
                               ws_priors={'mu': -1, 'sd': 2},
                               lambda_priors={'mu': 0, 'sd': 1},
                               gam_priors={'mu': 0, 'sd': 1},
                               ndt_priors={'mu': 0, 'sd': 2})
        else:
            self.priors = dict(sigma_priors={'mu_mu': 1, 'sd_mu': 3, 'mu_sd': 0, 'sd_sd': 3},
                               v_priors={'mu_mu': 9, 'sd_mu': 3, 'mu_sd': 2, 'sd_sd': 2},
                               wd_priors={'mu_mu': 1, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1},
                               ws_priors={'mu_mu': -1, 'sd_mu': 2, 'mu_sd': 3, 'sd_sd': 1},
                               lambda_priors={'mu_mu': 0, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1},
                               gam_priors={'mu_mu': 0, 'sd_mu': 1, 'mu_sd': 1, 'sd_sd': 1},
                               ndt_priors={'mu_mu': 0, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1})

        # Set up model label and priors for mechanisms
        if not self.additive and not self.multiplicative: # pure magnitude model
            self.model_label += '_pure_value'
            self.priors.pop('lambda_priors', None)
            self.priors.pop('gam_priors', None)
        elif self.additive and not self.multiplicative: # pure additive model
            self.model_label += '_additive'
            self.n_parameters_individual += 1 # lambda
            self.priors.pop('gam_priors', None)
        elif not self.additive and self.multiplicative: # pure multiplicative model
            self.model_label += '_multiplicative'
            self.priors.pop('lambda_priors', None)
            self.n_parameters_individual += 1 # gamma
        elif self.additive and self.multiplicative: # full GARD
            self.model_label += '_hybrid'
            self.n_parameters_individual += 2 # gamma, lambda
        # elif self.model_kind == 'model4': # sigma
        #     self.model_label += '_model4'
        #     self.priors.pop('ws_priors', None)
        #     self.n_parameters_individual += 1 # gamma, lambda

        # Set the stan model path
        self._set_model_path()

        # Finally, compile the model
        self._compile_stan_model()

    def fit(self,
            data,
            sigma_priors=None,
            wd_priors=None,
            ws_priors=None,
            v_priors=None,
            include_rhat=True,
            include_waic=True,
            pointwise_waic=False,
            include_last_values=True,
            print_diagnostics=True,
            **kwargs):
       
        data.reset_index(inplace=True)
        N = data.shape[0]  # n observations

        data_dict = {'N': N,
                     'rt': data['rt'].values,
                     'choice': data['choice'].values.astype(int),
                     'G1': data['gaze_1'].values.astype(float),
                     'G2': data['gaze_2'].values.astype(float),
                     'val1': data['item_value_1'].values,
                     'val2': data['item_value_2'].values}

        # change default priors:
        if sigma_priors is not None:
            self.priors['sigma_priors'] = sigma_priors
        if v_priors is not None:
            self.priors['v_priors'] = v_priors
        if wd_priors is not None:
            self.priors['wd_priors'] = wd_priors
        if ws_priors is not None:
            self.priors['ws_priors'] = ws_priors

        if self.hierarchical_levels == 2:
            keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
            L = len(pd.unique(data.sbj))  # n subjects (levels)
            data_dict.update({'L': L,
                              'sbj': data['sbj'].values.astype(int)})
        else:
            keys_priors = ["mu", "sd"]

        # Add data for mechanisms:

        # Add priors:
        print("Fitting the model using the priors:")
        for par in self.priors.keys():
            data_dict.update({par: [self.priors[par][key] for key in keys_priors]})
            print(par, self.priors[par])

        # start sampling...
        fitted_model = self.compiled_model.sample(data_dict, **kwargs)

        fitted_model = RDMFittedModel_2A(fitted_model,
                                         data,
                                         self.hierarchical_levels,
                                         self.model_label,
                                         self.family,
                                         self.n_parameters_individual,
                                         self.n_parameters_trial,
                                         print_diagnostics,
                                         self.priors,
                                         False)

        res = fitted_model.extract_results(include_rhat,
                                           include_waic,
                                           pointwise_waic,
                                           include_last_values)

        return res


class GARDModel_nA(Model):
    def __init__(self, hierarchical_levels, 
                 additive=True,
                 multiplicative=True):
        super().__init__(hierarchical_levels, "GARD_nA")

        self.additive = additive
        self.multiplicative = multiplicative

        # Define the model parameters
        self.n_parameters_individual = 5  # v, wd, ws, sigma, ndt
        self.n_parameters_trial = 0

        # Define default priors
        if self.hierarchical_levels == 1:
            self.priors = dict(v_priors={'mu': 9, 'sd': 3},
                               wd_priors={'mu': 1, 'sd': 2},
                               ws_priors={'mu': -1, 'sd': 2},
                               sigma_priors={'mu': 0, 'sd': 3},
                               lambda_priors={'mu': 0, 'sd': 1},
                               gam_priors={'mu': 0, 'sd': 1},
                               ndt_priors={'mu': 0, 'sd': 2})
        else:
            self.priors = dict(v_priors={'mu_mu': 9, 'sd_mu': 3, 'mu_sd': 2, 'sd_sd': 2},
                               wd_priors={'mu_mu': 1, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1},
                               ws_priors={'mu_mu': -1, 'sd_mu': 2, 'mu_sd': 3, 'sd_sd': 1},
                               sigma_priors={'mu_mu': 1, 'sd_mu': 3, 'mu_sd': 0, 'sd_sd': 3},
                               lambda_priors={'mu_mu': 0, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1},
                               gam_priors={'mu_mu': 0, 'sd_mu': 1, 'mu_sd': 1, 'sd_sd': 1},
                               ndt_priors={'mu_mu': 0, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1})

        # Set up model label and priors for mechanisms
        if not self.additive and not self.multiplicative: # pure magnitude model
            self.model_label += '_pure_value'
            self.priors.pop('lambda_priors', None)
            self.priors.pop('gam_priors', None)
        elif self.additive and not self.multiplicative: # pure additive model
            self.model_label += '_additive'
            self.n_parameters_individual += 1 # lambda
            self.priors.pop('gam_priors', None)
        elif not self.additive and self.multiplicative: # pure multiplicative model
            self.model_label += '_multiplicative'
            self.priors.pop('lambda_priors', None)
            self.n_parameters_individual += 1 # gamma
        elif self.additive and self.multiplicative: # full GARD
            self.model_label += '_hybrid'
            self.n_parameters_individual += 2 # gamma, lambda
        # elif self.model_kind == 'model4': # sigma
        #     self.model_label += '_model4'
        #     self.priors.pop('ws_priors', None)
        #     self.n_parameters_individual += 1 # gamma, lambda

        # Set the stan model path
        self._set_model_path()

        # Finally, compile the model
        self._compile_stan_model()

    def fit(self,
            data,
            v_priors=None,
            wd_priors=None,
            ws_priors=None,
            sigma_priors=None,
            include_rhat=True,
            include_waic=True,
            pointwise_waic=False,
            include_last_values=True,
            print_diagnostics=False,
            **kwargs):
       
        data.reset_index(inplace=True)
        N = data.shape[0]  # n observations

        data_dict = {'N': N,
                     'rt': data['rt'].values,
                     'choice': data['choice'].values.astype(int),
                     'G': data[['gaze_0', 'gaze_1', 'gaze_2']].values.astype(float),
                     'val': data[['item_value_0', 'item_value_1', 'item_value_2']].values}

        # change default priors:
        if v_priors is not None:
            self.priors['v_priors'] = v_priors
        if wd_priors is not None:
            self.priors['wd_priors'] = wd_priors
        if ws_priors is not None:
            self.priors['ws_priors'] = ws_priors
        if sigma_priors is not None:
            self.priors['sigma_priors'] = sigma_priors
        
        if self.hierarchical_levels == 2:
            keys_priors = ["mu_mu", "sd_mu", "mu_sd", "sd_sd"]
            L = len(pd.unique(data.sbj))  # n subjects (levels)
            data_dict.update({'L': L,
                              'sbj': data['sbj'].values.astype(int)})
        else:
            keys_priors = ["mu", "sd"]

        # Add data for mechanisms:

        # Add priors:
        print("Fitting the model using the priors:")
        for par in self.priors.keys():
            data_dict.update({par: [self.priors[par][key] for key in keys_priors]})
            print(par, self.priors[par])

        # start sampling...
        fitted_model = self.compiled_model.sample(data_dict, **kwargs)

        fitted_model = RDMFittedModel_nA(fitted_model,
                                         3,
                                         data,
                                         self.hierarchical_levels,
                                         self.model_label,
                                         self.family,
                                         self.n_parameters_individual,
                                         self.n_parameters_trial,
                                         print_diagnostics,
                                         self.priors)

        res = fitted_model.extract_results(include_rhat,
                                           include_waic,
                                           pointwise_waic,
                                           include_last_values)

        return res
