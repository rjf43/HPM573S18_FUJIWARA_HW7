import Model as Model

# Create a calibration object
calibration = Model.Calibration()

# Sample the posterior of the mortality probability
calibration.sample_posterior()

# Estimate of mortality probability and the posterior interval
print(calibration.get_mortality_estimate_credible_interval(Model.ALPHA, 4))