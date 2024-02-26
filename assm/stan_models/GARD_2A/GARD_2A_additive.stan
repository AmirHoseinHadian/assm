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
          cdf = 1 - race_cdf(t, sigma[i], drift_right[i]);
        }
        else{
          pdf = race_pdf(t, sigma[i], drift_right[i]);
          cdf = 1 - race_cdf(t, sigma[i], drift_left[i]);
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
  int<lower=1,upper=2> choice[N];
  real<lower=0> rt[N];							// rt

  real<lower=0> G1[N];
  real<lower=0> G2[N];

  real<lower=0> val1[N];
  real<lower=0> val2[N];

  vector[2] v_priors;
  vector[2] wd_priors;
  vector[2] ws_priors;
  vector[2] sigma_priors;
  vector[2] lambda_priors;
  vector[2] ndt_priors;
}

transformed data {
  matrix [N, 2] RT;

  for (n in 1:N){
    RT[n, 1] = rt[n];
    RT[n, 2] = choice[n];
  }
}

parameters {
  real v;
  real wd;
  real ws;
  real sigma;
  real lambda;
  real ndt;
}

transformed parameters {
  vector[N] drift_left_t;				// trial-by-trial drift rate for predictions
  vector[N] drift_right_t;				// trial-by-trial drift rate for predictions
  vector<lower=0>[N] sigma_t;		// trial-by-trial sigma
  vector<lower=0>[N] ndt_t;

  real<lower=0> transf_v;
  real<lower=0> transf_wd;
  real<lower=0> transf_ws;
  real<lower=0> transf_sigma;
  real<lower=0> transf_lambda;
  real<lower=0> transf_ndt;

  transf_v = log(1 + exp(v));
  transf_wd = log(1 + exp(wd));
  transf_ws = log(1 + exp(ws));
  transf_sigma = log(1 + exp(sigma));
  transf_lambda = log(1 + exp(lambda));
  transf_ndt = log(1 + exp(ndt));

  for (n in 1:N) {
    sigma_t[n] = transf_sigma;
    ndt_t[n] = transf_ndt;

    drift_left_t[n] = transf_v + transf_wd*(val1[n] - val2[n]) + transf_ws*(val1[n] + val2[n]) + transf_lambda*G1[n];
    drift_right_t[n] = transf_v + transf_wd*(val2[n] - val1[n]) + transf_ws*(val2[n] + val1[n]) + transf_lambda*G2[n];
  }
}

model {
  v ~ normal(v_priors[1], v_priors[2]);
  wd ~ normal(wd_priors[1], wd_priors[2]);
  ws ~ normal(ws_priors[1], ws_priors[2]);
  sigma ~ normal(sigma_priors[1], sigma_priors[2]);
  lambda ~ normal(lambda_priors[1], lambda_priors[2]);
  ndt ~ normal(ndt_priors[1], ndt_priors[2]);

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
