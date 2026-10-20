import pandas as pd


# Pair correlation
cor = pd.read_csv("/run/media/thomas/Data/Documents/Database/results/full/correlation/spearman/top_pairs_cross_feature.csv")

harmonic_rms_a = cor['column_a'].str.contains('harmonic_rms')
harmonic_ratio_a = cor['column_a'].str.contains('harmonic_ratio')
percussive_ratio_a = cor['column_a'].str.contains('percussive_ratio')
tonnetz_cqt_a = cor['column_a'].str.contains('tonnetz_cqt')
tonnetz_cens_a = cor['column_a'].str.contains('tonnetz_cens')
beat_interval_a = cor['column_a'].str.contains('beat_interval')
tempo_a = cor['column_a'].str.contains('tempo')
beat_rate_a = cor['column_a'].str.contains('beat_rate')
rms_a = cor['column_a'].str.contains('rms')

harmonic_rms_b = cor['column_b'].str.contains('harmonic_rms')
harmonic_ratio_b = cor['column_b'].str.contains('harmonic_ratio')
percussive_ratio_b = cor['column_b'].str.contains('percussive_ratio')
tonnetz_cqt_b = cor['column_b'].str.contains('tonnetz_cqt')
tonnetz_cens_b = cor['column_b'].str.contains('tonnetz_cens')
beat_interval_b = cor['column_b'].str.contains('beat_interval')
tempo_b = cor['column_b'].str.contains('tempo')
beat_rate_b = cor['column_b'].str.contains('beat_rate')
rms_b = cor['column_b'].str.contains('rms')

print(cor[harmonic_rms_a])

# Pruning by feature
cor = pd.read_csv("/run/media/thomas/Data/Documents/Database/results/full/correlation/spearman/pruning_by_feature.csv").sort_values('pct_kept')
print(cor)


# Redundancy
cor = pd.read_csv("/run/media/thomas/Data/Documents/Database/results/full/correlation/spearman/feature_redundancy.csv")

