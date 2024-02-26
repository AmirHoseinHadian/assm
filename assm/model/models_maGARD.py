from __future__ import absolute_import, division, print_function
import pandas as pd

from assm.fit.fits_RDM import RDMFittedModel_2A
from assm.model.models import Model


class maGARDModel_2A(Model):
    def __init__(self, hierarchical_levels, 
                 additive=True,
                 multiplicative=True,
                 attribute_attention=True,
                 fix_w=False):
        super().__init__(hierarchical_levels, "maGARD_2A")

        self.additive = additive
        self.multiplicative = multiplicative
        self.attribute_attention = attribute_attention
        self.fix_w = fix_w
        
        
        # Define the model parameters
        self.n_parameters_individual = 5  # sigma, v, wd1, wd2, ndt
        self.n_parameters_trial = 0

        # Define default priors
        if self.hierarchical_levels == 1:
            self.priors = dict(sigma_priors={'mu': 0, 'sd': 3},
                               v_priors={'mu': 7, 'sd': 3},
                               wd_priors={'mu': 0, 'sd': 2},
                            #    ws_priors={'mu': -1, 'sd': 2},
                               lambda_priors={'mu': 0, 'sd': 3},
                               gam_priors={'mu': 0, 'sd': 1},
                               ndt_priors={'mu': 0, 'sd': 2}
                            #    omg_priors={'mu': 0, 'sd': 2}
                               )
        else:
            self.priors = dict(sigma_priors={'mu_mu': 1, 'sd_mu': 3, 'mu_sd': 0, 'sd_sd': 3},
                               v_priors={'mu_mu': 7, 'sd_mu': 3, 'mu_sd': 2, 'sd_sd': 2},
                               wd_priors={'mu_mu': 1, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1},
                            #    ws_priors={'mu_mu': -1, 'sd_mu': 2, 'mu_sd': 3, 'sd_sd': 1},
                               lambda_priors={'mu_mu': 0, 'sd_mu': 3, 'mu_sd': 2, 'sd_sd': 1},
                               gam_priors={'mu_mu': 0, 'sd_mu': 1, 'mu_sd': 1, 'sd_sd': 1},
                               ndt_priors={'mu_mu': 0, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1}
                            #    omg_priors={'mu_mu': 0, 'sd_mu': 2, 'mu_sd': 1, 'sd_sd': 1}
                               )
                               
        
        if not self.attribute_attention:
            # Set up model label and priors for mechanisms
            if self.additive and not self.multiplicative: 
                self.model_label += '_additive'
                self.priors.pop('gam_priors', None)
                self.n_parameters_individual += 1 # lambda
            elif not self.additive and self.multiplicative:
                self.model_label += '_multiplicative'
                self.priors.pop('lambda_priors', None)
                self.n_parameters_individual += 2 # gamma_op, gamma_at
            elif self.additive and self.multiplicative:
                self.model_label += '_hybrid'
                self.n_parameters_individual += 3 # gamma_op, gamma_at, lambda
        else:
            if self.additive and not self.multiplicative: 
                self.model_label += '_additive_atatt'
                self.priors.pop('gam_priors', None)
                self.n_parameters_individual += 2 # lambda1, lambda2
            elif self.additive and self.multiplicative:
                if self.fix_w:
                    self.model_label += '_hybrid_atatt_fw'
                    self.n_parameters_individual += 3 # gamma_op, gamma_at, lambda1, lambda2 (wd1 = wd2)
                else:
                    self.model_label += '_hybrid_atatt'
                    self.n_parameters_individual += 4 # gamma_op, gamma_at, lambda1, lambda2

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
                     'G_o1_a1': data['o1_a1_gaze'].values.astype(float),
                     'G_o1_a2': data['o1_a2_gaze'].values.astype(float),
                     'G_o2_a1': data['o2_a1_gaze'].values.astype(float),
                     'G_o2_a2': data['o2_a2_gaze'].values.astype(float),
                     'val_o1_a1': data['o1_a1_val'].values,
                     'val_o1_a2': data['o1_a2_val'].values,
                     'val_o2_a1': data['o2_a1_val'].values,
                     'val_o2_a2': data['o2_a2_val'].values}

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

