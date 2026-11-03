import numpy as np
import pandas as pd
import utils


X_small = utils.load("/run/media/thomas/Data/Documents/MusiClassi//fma_metadata/myfeatures_small.csv")
X_medium = utils.load("/run/media/thomas/Data/Documents/MusiClassi//fma_metadata/myfeatures_medium.csv")

# Check features DF is the same
assert X_small.columns.equals(X_medium.columns)
assert X_small.columns.is_unique
assert X_medium.columns.is_unique

# Missing and infinite values
for name, X in [
    ("FMA_small", X_small),
    ("FMA_medium", X_medium),
]:
    print(f"\n{name}")
    print("Shape:", X.shape)
    print("Infinite values:", np.isinf(X.to_numpy()).sum())
    print("NaN values:", X.isna().sum().sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    failed = X.index[X.isna().all(axis=1)]
    print(f"Tracks with NaN or Inf values: {len(failed)}")
