import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

r = np.random.default_rng(7).normal(0.0004, 0.01, 125)
s = pd.Series(r)
print("pandas", pd.__version__, "scipy", scipy.__version__)
print("skew    ", repr(float(s.skew())))
print("kurt    ", repr(float(s.kurtosis())))
print("std     ", repr(float(s.std())))
print("norm.cdf", repr(float(norm.cdf(0.123456))), "ppf", repr(float(norm.ppf(0.987654321))))
