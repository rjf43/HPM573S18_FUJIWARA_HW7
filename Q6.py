from enum import Enum
import scipy.stats as stat
import numpy as np
import Q1 as SurvivalCls
import scr.StatisticalClasses as StatSupport
import scr.FormatFunctions as FormatSupport
import scr.InOutFunctions as InOutSupport

TIME_STEPS = 1000
ALPHA = 0.05
NUM_SIM_COHORTS = 500

# details of clinical study
STUDY_K = 800
STUDY_N = 1146

# details of projected posterior
POST_L = 0.01
POST_U = 0.30
POST_N = 1000

class CalibrationColIndex(Enum):
    ID = 0
    W = 1
    MORT_PROB = 2

class Calibration:
    def __init__(self):
        np.random.seed(1)
        self._cohortIDs = range(POST_N)
        self._mortalitySamples = []
        self._mortalityResamples = []
        self._weights = []
        self._normalizedWeights = []
        self._csvRows = \
            [['Cohort ID', 'Likelihood Weights', 'Mortality Prob']]

    def sample_posterior(self):
        """ sample the posterior distribution of the mortality probability """

        # find values of mortality probability at which the posterior should be evaluated
        self._mortalitySamples = np.random.uniform(
            low = POST_L,
            high = POST_U,
            size = POST_N
        )

        # create a multi cohort
        multiCohort = SurvivalCls.MultiCohort(
            ids= self._cohortIDs,
            pop_sizes= [POST_N]*POST_N,
            mortality_probs= self._mortalitySamples
        )

        # simulate the multi cohort
        multiCohort.simulate(TIME_STEPS)

        # calculate the likelihood of each simulated cohort
        for i in self._cohortIDs:

            # get the 5-year OS for this cohort
            survival = multiCohort.get_cohort_FIVEyear_OS(i)

            # construct weight utilizing study's k and n; and simulated five-year OS
            weight = stat.binom.pmf(
                k = STUDY_K,
                n = STUDY_N,
                p = survival
            )

            # store the weight
            self._weights.append(weight)

        # normalize the likelihood weights
        sum_weights = np.sum(self._weights)
        self._normalizedWeights = np.divide(self._weights, sum_weights)

        # re-sample mortality probability (with replacement) according to likelihood weights
        self._mortalityResamples = np.random.choice(
            a = self._mortalitySamples,
            size = NUM_SIM_COHORTS,
            replace = True,
            p = self._normalizedWeights
        )

        # produce the list to report the results
        for i in range(0, len(self._mortalitySamples)):
            self._csvRows.append(
                [self._cohortIDs[i], self._normalizedWeights[i], self._mortalitySamples[i]]
            )

        # write the calibration result into a csv file
        InOutSupport.write_csv('CalibrationResults.csv', self._csvRows)

    def get_mortality_resamples(self):
        return self._mortalityResamples()

    def get_mortality_estimate_credible_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :return: text in the form of 'mean (lower, upper)' of the posterior distribution
        """

        # calculate the credible interval
        sum_stat = StatSupport.SummaryStat('Posterior samples', self._mortalityResamples)
        estimate = sum_stat.get_mean() # estimated mortality probability
        credible_interval = sum_stat.get_PI(alpha) # credible interval

        return FormatSupport.format_estimate_interval(estimate, credible_interval, deci)

class CalibratedModel:
    """ to run the calibrated survival model """

    def __init__(self, csv_file_name):

        # read the columns of the generated csv file containing the calibration results
        cols = InOutSupport.read_csv_cols(
            file_name= csv_file_name,
            n_cols= 3,
            if_ignore_first_row= True,
            if_convert_float= True
        )

        # store cohort IDs, likelihood weights, and mortality probabilities
        self._cohortIDs = cols[CalibrationColIndex.ID.value].astype(int)
        self._weights = cols[CalibrationColIndex.W.value]
        self._mortalityProbs = cols[CalibrationColIndex.MORT_PROB.value]
        self._multiCohorts = None

    def simulate(self, num_simulated_cohorts, cohort_size, time_steps):
        """
        :param num_simulated_cohorts: number of cohorts to simulate
        :param cohort_size: population size of cohorts
        :param time_steps: simulation length
        :param cohort_ids: ids of cohort to simulate
        :return:
        """

        # resample cohort IDs and mortality probabilities based on weights
        sampled_row_indices = np.random.choice(
            a= range(0, len(self._weights)),
            size= num_simulated_cohorts,
            replace= True,
            p= self._weights
        )

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        resampled_IDs = []
        resampled_mortalityprobs = []
        for i in sampled_row_indices:
            resampled_IDs.append(self._cohortIDs[i])
            resampled_mortalityprobs.append(self._mortalityProbs[i])

        # simulate the desired number of cohorts
        self._multiCohorts = SurvivalCls.MultiCohort(
            ids= resampled_IDs,
            pop_sizes= [cohort_size]*num_simulated_cohorts,
            mortality_probs= resampled_mortalityprobs
        )

        # simulate all the cohorts
        self._multiCohorts.simulate(time_steps)

    def get_all_mean_survival(self):
        return self._multiCohorts.get_all_mean_survival()

    def get_mean_survival_time_proj_interval(self, alpha, deci):
        mean = self._multiCohorts.get_overall_mean_survival()
        proj_interval = self._multiCohorts.get_PI_mean_survival(alpha)

        return FormatSupport.format_estimate_interval(mean, proj_interval, deci)

# MORTALITY PROBABILITY AND CREDIBLE INTERVAL
calibration = Calibration() # calibrate
calibration.sample_posterior() # Sample the posterior of the mortality probability
print(calibration.get_mortality_estimate_credible_interval(ALPHA, 4)) # Estimate of mortality probability and the posterior interval

# MEAN SURVIVAL TIME AND PROJECTION INTERVAL
calibrated_model = CalibratedModel('CalibrationResults.csv') # calibrate the model
calibrated_model.simulate(NUM_SIM_COHORTS, POST_N, TIME_STEPS) # simulate the calibrated model
print(calibrated_model.get_mean_survival_time_proj_interval(ALPHA,4))

print('The half-length of the credible interval and the projection interval both decrease significantly as a result of increased sample size in the sample population.')

