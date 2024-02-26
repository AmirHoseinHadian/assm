functions {
  real race_pdf(real t, real sigma, real v){
    real pdf;
    real lambda, mu;
    mu = 1/v;
    lambda = 1/pow(sigma, 2);
    pdf = sqrt(lambda/(2 * pi() * pow(t, 3))) * exp(-(lambda * pow(t-mu, 2)) / (2*pow(mu, 2)*t));
    return pdf;
  }

  real race_cdf(real t, real sigma, real v){
    real cdf;
    real lambda, mu;
    mu = 1/v;
    lambda = 1/pow(sigma, 2);
    cdf = Phi(sqrt(lambda/t)*(t/mu - 1)) + exp(2*lambda/mu) * Phi(-sqrt(lambda/t)*(t/mu + 1));
    return cdf;
  }

  real race_lpdf(matrix RT, vector sigma, vector ndt, vector drift_left, vector drift_right){

    real t;
    vector[rows(RT)] prob;
    real cdf;
    real pdf;
    real out;

    for (i in 1:rows(RT)){
      t = RT[i,1] - ndt[i];
      if(t > 0){
        if(RT[i,2] == 1){
          pdf = race_pdf(t, sigma[i], drift_left[i]);
          cdf = 1 - race_cdf(t| sigma[i], drift_right[i]);
        }
        else{
          pdf = race_pdf(t, sigma[i], drift_right[i]);
          cdf = 1 - race_cdf(t| sigma[i], drift_left[i]);
        }
        prob[i] = pdf*cdf;
        if(prob[i] < 1e-10 ){
            prob[i] = 1e-10;
        }
        if(is_nan(prob[i])){
          prob[i] = 1e-10;
        }
      }
      else{
          prob[i] = 1e-10;
      }
    }
    out = sum(log(prob));
    return out;
  }
}

data {
  int<lower=1> N;									// number of data items
  int<lower=1> L;									// number of levels
  array[N] int<lower=1, upper=L> sbj;			// level (participant)

  array[N] int<lower=1,upper=2> choice;
  array[N] real<lower=0> rt;							// rt

  array[N] real<lower=0> G1;
  array[N] real<lower=0> G2;

  array[N] real<lower=0> val1;
  array[N] real<lower=0> val2;

  vector[4] v_priors;
  vector[4] wd_priors;
  vector[4] ws_priors;
  vector[4] sigma_priors;
  vector[4] lambda_priors;
  vector[4] gam_priors;
  vector[4] ndt_priors;
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
    RT[n, 1] = rt[n];
    RT[n, 2] = choice[n];
	}
}

parameters {
  real mu_v;
  real mu_wd;
  real mu_ws;
  real mu_sigma;
  real mu_lambda;
  real mu_gam;
  real mu_ndt;

  real<lower=0> sd_v;
  real<lower=0> sd_wd;
  real<lower=0> sd_ws;
  real<lower=0> sd_sigma;
  real<lower=0> sd_lambda;
  real<lower=0> sd_gam;
  real<lower=0> sd_ndt;

  array[L] real z_v;
  array[L] real z_wd;
  array[L] real z_ws;
  array[L] real z_sigma;
  array[L] real z_lambda;
  array[L] real z_gam;
  array[L] real z_ndt;
}

transformed parameters {
  vector[N] drift_left_t;				// trial-by-trial drift rate for predictions
  vector[N] drift_right_t;				// trial-by-trial drift rate for predictions
  vector<lower=0>[N] sigma_t;		// trial-by-trial sigma
  vector<lower=0>[N] ndt_t;

  array[L] real<lower=0> v_sbj;
  array[L] real<lower=0> wd_sbj;
  array[L] real<lower=0> ws_sbj;
  array[L] real<lower=0> sigma_sbj;
  array[L] real<lower=0> lambda_sbj;
  array[L] real<lower=0, upper=1> gam_sbj;
  array[L] real<lower=0> ndt_sbj;

  real<lower=0> transf_mu_v;
  real<lower=0> transf_mu_wd;
  real<lower=0> transf_mu_ws;
  real<lower=0> transf_mu_sigma;
  real<lower=0> transf_mu_lambda;
  real<lower=0, upper=1> transf_mu_gam;
  real<lower=0> transf_mu_ndt;

  real A1, A2;

  transf_mu_v = log(1 + exp(mu_v));
  transf_mu_wd = log(1 + exp(mu_wd));
  transf_mu_ws = log(1 + exp(mu_ws));
  transf_mu_sigma = log(1 + exp(mu_sigma));
  transf_mu_lambda = log(1 + exp(mu_lambda));
  transf_mu_gam = Phi(mu_gam);
  transf_mu_ndt = log(1 + exp(mu_ndt));
  
	for (l in 1:L) {
    v_sbj[l] = log(1 + exp(mu_v + z_v[l]*sd_v));
    wd_sbj[l] = log(1 + exp(mu_wd + z_wd[l]*sd_wd));
    ws_sbj[l] = log(1 + exp(mu_ws + z_ws[l]*sd_ws));
    sigma_sbj[l] = log(1 + exp(mu_sigma + z_sigma[l]*sd_sigma));
    lambda_sbj[l] = log(1 + exp(mu_lambda + z_lambda[l]*sd_lambda));
    gam_sbj[l] = Phi(mu_gam + z_gam[l]*sd_gam);
    ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
    sigma_t[n] = sigma_sbj[sbj[n]];
    ndt_t[n] = ndt_sbj[sbj[n]];
    
    A1 = G1[n]*val1[n] + gam_sbj[sbj[n]]*(1 - G1[n])*val1[n];
    A2 = G2[n]*val2[n] + gam_sbj[sbj[n]]*(1 - G2[n])*val2[n];

    drift_left_t[n] = v_sbj[sbj[n]] + wd_sbj[sbj[n]]*(A1 - A2) + ws_sbj[sbj[n]]*(A1 + A2) + lambda_sbj[sbj[n]]*G1[n];
    drift_right_t[n] = v_sbj[sbj[n]] + wd_sbj[sbj[n]]*(A2 - A1) + ws_sbj[sbj[n]]*(A2 + A1) + lambda_sbj[sbj[n]]*G2[n];
  }
}

model {
  mu_v ~ normal(v_priors[1], v_priors[2]);
  mu_wd ~ normal(wd_priors[1], wd_priors[2]);
  mu_ws ~ normal(ws_priors[1], ws_priors[2]);
  mu_sigma ~ normal(sigma_priors[1], sigma_priors[2]);
  mu_lambda ~ normal(lambda_priors[1], lambda_priors[2]);
  mu_gam ~ normal(gam_priors[1], gam_priors[2]);
  mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

  sd_v ~ normal(v_priors[3], v_priors[4]);
  sd_wd ~ normal(wd_priors[3], wd_priors[4]);
  sd_ws ~ normal(ws_priors[3], ws_priors[4]);
  sd_sigma ~ normal(sigma_priors[3], sigma_priors[4]);
  sd_lambda ~ normal(lambda_priors[3], lambda_priors[4]);
  sd_gam ~ normal(gam_priors[3], gam_priors[4]);
  sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

  z_v ~ normal(0, 1);
  z_wd ~ normal(0, 1);
  z_ws ~ normal(0, 1);
  z_sigma ~ normal(0, 1);
  z_lambda ~ normal(0, 1);
  z_gam ~ normal(0, 1);
  z_ndt ~ normal(0, 1);

  RT ~ race(sigma_t, ndt_t, drift_left_t, drift_right_t);
}

generated quantities {
  vector[N] log_lik;
  {
    for (n in 1:N){
      log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(sigma_t, n, 1), segment(ndt_t, n, 1), segment(drift_left_t, n, 1), segment(drift_right_t, n, 1));
    }
  }
}
