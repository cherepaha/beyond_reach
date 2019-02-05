get_bf_dyn <- function(models, data, formulas, prior){ 
  " This function fits three models to the supplied data, and returns these models with pairwise BF's"
  m_delta <- fit_model(models$m_delta, data, formulas$f_delta, prior)
  m_inter <- fit_model(models$m_inter, data, formulas$f_inter, prior)
  m_choice <- fit_model(models$m_choice, data, formulas$f_choice, prior[1,])
  m_null <- fit_model(models$m_null, data, formulas$f_null, prior[1,])
  
  bf <- t(c(bayes_factor(x1 = m_choice, x2 = m_null)$bf,
            bayes_factor(x1 = m_inter, x2 = m_null)$bf,
            bayes_factor(x1 = m_inter, x2 = m_choice)$bf,
            bayes_factor(x1 = m_delta, x2 = m_inter)$bf))
  
  result = list(bf = bf, models = list(m_null = m_null, m_choice = m_choice, m_inter = m_inter, m_delta = m_delta))
  
  return(result)
}

fit_model <- function(model, data, formula, prior){
  " This function updates the brms model if it exists (to avoid slow recompilation of the C++ model by Stan), 
  or creates a model from scratch if it doesn't exist yet "
  if(!is.null(model)){
    model <- update(model, newdata = data, recompile = F)        
  } else {
    model <- brm(formula, data = data, family = gaussian(), save_all_pars = TRUE,
                 control = list(adapt_delta = 0.99), prior = prior)
  }
  return(model)
}

run_analysis <- function(dv, iv, data, rscale){
  " This function runs our analysis for a given dataset (fake or real) and given dependent and 
  independent variables (dv and iv). Tscale parameter defines the scale of prior distribution on regression slopes (see Rouder & Morey (2012))"
  models <- list(m_delta=NULL, m_inter = NULL, m_choice = NULL, m_null = NULL)
  
  " Priors on centered intercepts are the values of DV when all predictors are at their means (see brms docs).
  brms accepts priors on non-standardized b coefficients. We compute them by scaling default priors on standardized 
  coefficients (Rouder & Morey, 2012) to sd(DV)/sd(IV)"
  prior <- c(set_prior(sprintf('normal(%f, %f)', mean(data[, dv]), sd(data[, dv])), class = "Intercept"),
             set_prior(sprintf('cauchy(0, %f)', rscale*sd(data[, dv])/sd(data[, iv])), class = "b"))
  
  interaction_term = paste(iv, "option_chosen", sep=':')
  formulas <- list(f_delta = as.formula(paste(dv, "~ (option_chosen | subj_id) + Delta +", interaction_term)),
                   f_inter = as.formula(paste(dv, "~ (option_chosen | subj_id) +", interaction_term)),
                   f_choice = as.formula(paste(dv, "~ (option_chosen | subj_id)")),
                   f_null = as.formula(paste(dv, "~ (1 | subj_id)")))
  
  bf_result <- get_bf_dyn(models, data, formulas, prior)
  
  return(bf_result)
}