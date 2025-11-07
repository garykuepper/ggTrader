from ggTrader.Utils import *
import pandas as pd
# 1. Create a sample time series dataset
# n_samples = 4057
X = pd.date_range(start='2023-01-01', end='2025-09-30', freq='4h')
n_samples = (len(X))

# 2. Configure end-anchored sliding window with variable ratio
n_splits = 5
test_ratio = 0.2  # 20% of each sliding window for testing

tscv, test_size, max_train_size = make_end_anchored_tscv(n_samples, n_splits, test_ratio)
print(f"Computed test_size={test_size}, max_train_size={max_train_size}")

# 3. Visualize the splits
fig, ax = plt.subplots(figsize=(10, 5))
plot_cv_indices(tscv, X, ax, n_splits)
plt.show()

# 4. Print fold ranges
for i, (tr, tt) in enumerate(tscv.split(X), 1):
    print(f"Fold {i}: Train [{X[tr[0]]}–{X[tr[-1]]}] ({len(tr)/6:.0f}), Test [{X[tt[0]]}–{X[tt[-1]]}] ({len(tt)/6:.0f})")
