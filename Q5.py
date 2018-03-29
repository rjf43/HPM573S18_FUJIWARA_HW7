import Model as Model

# Startup the calibrated model
calibrated_model = Model.CalibratedModel('CalibrationResults.csv')

# simulate the calibrated model
calibrated_model.simulate(Model.NUM_SIM_COHORTS, Model.POST_N, Model.TIME_STEPS)

# report mean and 95% projection interval
print(calibrated_model.get_mean_survival_time_proj_interval(Model.ALPHA,4))

