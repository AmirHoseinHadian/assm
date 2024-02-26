data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	int<lower=1, upper=L> sbj[N];			// level (participant)

	int<lower=-1,upper=1> choice[N];				// accuracy (-1, 1)
	real<lower=0> rt[N];							// rt

	real<lower=0> G1[N];
	real<lower=0> G2[N];

	real<lower=0> val1[N];
	real<lower=0> val2[N];

	vector[4] theta_priors;							// mean and sd of the prior
	vector[4] drift_scaling_priors;					// mean and sd of the prior
	vector[4] threshold_priors;						// mean and sd of the prior
	vector[4] ndt_priors;							// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}

parameters {
	real mu_theta;
	real mu_drift_scaling;
	real mu_threshold;
	real mu_ndt;

	real<lower=0> sd_theta;
	real<lower=0> sd_drift_scaling;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;

	real z_theta[L];
	real z_drift_scaling[L];
	real z_threshold[L];
	real z_ndt[L];
}

transformed parameters {
	real drift_ll[N];								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	real drift_t[N];								// trial-by-trial drift rate for predictions
	real<lower=0> threshold_t[N];					// trial-by-trial threshold
	real<lower=0> ndt_t[N];							// trial-by-trial ndt

	real<lower=0, upper=1> theta_sbj[L];
	real<lower=0> drift_scaling_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];

	real transf_mu_theta;
	real transf_mu_drift_scaling;
	real transf_mu_threshold;
	real transf_mu_ndt;

	transf_mu_theta = Phi(mu_theta);				// for the output
	transf_mu_drift_scaling = log(1 + exp(mu_drift_scaling));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
		theta_sbj[l] = Phi(mu_theta + z_theta[l]*sd_theta);
		drift_scaling_sbj[l] = log(1 + exp(mu_drift_scaling + z_drift_scaling[l]*sd_drift_scaling));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		drift_t[n] = drift_scaling_sbj[sbj[n]]*delta_Q[n];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = threshold_sbj[sbj[n]];
		ndt_t[n] = ndt_sbj[sbj[n]];
	}
}
model {
	mu_theta ~ normal(theta_priors[1], theta_priors[2]);
	mu_drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	sd_alpha ~ normal(theta_priors[3], theta_priors[4]);
	sd_drift_scaling ~ normal(drift_scaling_priors[3], drift_scaling_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

	z_alpha ~ normal(0, 1);
	z_drift_scaling ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
	}
}