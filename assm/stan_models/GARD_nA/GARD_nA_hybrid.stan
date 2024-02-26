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

     real race_lpdf(matrix RT, vector sigma, vector ndt, matrix drift){

          real t;
          vector[rows(RT)] prob;
          real t1, t2, t3, t4;
          real out;

          for (i in 1:rows(RT)){
               t = RT[i,1] - ndt[i];
               if(t > 0){
                if (RT[i,2] == 1){
                    t1 = race_pdf(t, sigma[i], drift[i, 1]) * race_cdf(t, sigma[i], drift[i, 2]);
                    t2 = race_pdf(t, sigma[i], drift[i, 2]) * race_cdf(t, sigma[i], drift[i, 1]);
                    t3 = (1 - race_cdf(t, sigma[i], drift[i, 3]) * race_cdf(t, sigma[i], drift[i, 4]));
                    t4 = (1 - race_cdf(t, sigma[i], drift[i, 5]) * race_cdf(t, sigma[i], drift[i, 6]));
                }
                if (RT[i,2] == 2){
                    t1 = race_pdf(t, sigma[i], drift[i, 3]) * race_cdf(t, sigma[i], drift[i, 4]);
                    t2 = race_pdf(t, sigma[i], drift[i, 4]) * race_cdf(t, sigma[i], drift[i, 3]);
                    t3 = (1 - race_cdf(t, sigma[i], drift[i, 1]) * race_cdf(t, sigma[i], drift[i, 2]));
                    t4 = (1 - race_cdf(t, sigma[i], drift[i, 5]) * race_cdf(t, sigma[i], drift[i, 6]));
                }
                if (RT[i,2] == 3){
                    t1 = race_pdf(t, sigma[i], drift[i, 5]) * race_cdf(t, sigma[i], drift[i, 6]);
                    t2 = race_pdf(t, sigma[i], drift[i, 6]) * race_cdf(t, sigma[i], drift[i, 5]);
                    t3 = (1 - race_cdf(t, sigma[i], drift[i, 1]) * race_cdf(t, sigma[i], drift[i, 2]));
                    t4 = (1 - race_cdf(t, sigma[i], drift[i, 3]) * race_cdf(t, sigma[i], drift[i, 4]));
                }
                prob[i] = (t1 + t2) * t3 * t4;
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

  int<lower=1,upper=3> choice[N];		
  real<lower=0> rt[N];							// rt

  real<lower=0> G[N, 3];							// rt

  real<lower=0> val[N, 3];							// 

  vector[2] v_priors;
  vector[2] wd_priors;
  vector[2] ws_priors;
  vector[2] sigma_priors;
  vector[2] lambda_priors;
  vector[2] gam_priors;
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
  real gam;
  real ndt;
}

transformed parameters {
  matrix[N, 6] drift_t;		// trial-by-trial drift rate for predictions
  vector<lower=0>[N] sigma_t;				// trial-by-trial threshold
  vector<lower=0>[N] ndt_t;				// trial-by-trial threshold

  real<lower=0> transf_v;
  real<lower=0> transf_wd;
  real<lower=0> transf_ws;
  real<lower=0> transf_sigma;
  real<lower=0> transf_lambda;
  real<lower=0, upper=1> transf_gam;
  real<lower=0> transf_ndt;

  real A[3];

  transf_v = log(1 + exp(v));
  transf_wd = log(1 + exp(wd));
  transf_ws = log(1 + exp(ws));
  transf_sigma = log(1 + exp(sigma));
  transf_lambda = log(1 + exp(lambda));
  transf_gam = Phi(gam);
  transf_ndt = log(1 + exp(ndt));

	for (n in 1:N) {
    sigma_t[n] = transf_sigma;
    ndt_t[n] = transf_ndt;

    for (i in 1:3) {
      A[i] = G[n, i]*val[n, i] + transf_gam*(1 - G[n, i])*val[n, i];
    }

    drift_t[n, 1] = transf_v + transf_wd*(A[1] - A[2]) + transf_ws*(A[1] + A[2]) + transf_lambda*G[n, 1];
    drift_t[n, 2] = transf_v + transf_wd*(A[1] - A[3]) + transf_ws*(A[1] + A[3]) + transf_lambda*G[n, 1];
    drift_t[n, 3] = transf_v + transf_wd*(A[2] - A[1]) + transf_ws*(A[2] + A[1]) + transf_lambda*G[n, 2];
    drift_t[n, 4] = transf_v + transf_wd*(A[2] - A[3]) + transf_ws*(A[2] + A[3]) + transf_lambda*G[n, 2];
    drift_t[n, 5] = transf_v + transf_wd*(A[3] - A[1]) + transf_ws*(A[3] + A[1]) + transf_lambda*G[n, 3];
    drift_t[n, 6] = transf_v + transf_wd*(A[3] - A[2]) + transf_ws*(A[3] + A[2]) + transf_lambda*G[n, 3];
  }
}

model {
  v ~ normal(v_priors[1], v_priors[2]);
  wd ~ normal(wd_priors[1], wd_priors[2]);
  ws ~ normal(ws_priors[1], ws_priors[2]);
  sigma ~ normal(sigma_priors[1], sigma_priors[2]);
  lambda ~ normal(lambda_priors[1], lambda_priors[2]);
  gam ~ normal(gam_priors[1], gam_priors[2]);
  ndt ~ normal(ndt_priors[1], ndt_priors[2]);

  RT ~ race(sigma_t, ndt_t, drift_t);
}

generated quantities {
	vector[N] log_lik;
	{
    for (n in 1:N){
      log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(sigma_t, n, 1), segment(ndt_t, n, 1), block(drift_t, n, 1, 1, 6));
    }
	}
}