import scipy.stats as stat

STUDY_K = 400
STUDY_N = 573
THETA = 0.50

likelihood = stat.binom.pmf(
    k=STUDY_K,
    n= STUDY_N,
    p= THETA,
)

print(likelihood)