# Description
A project that aims at classifying songs based upon their musical genres and generate playlists according to these genres.

## Ideas
- Compare CNN to classic ML algo
- Once a model perform ok on FMA data, try to increase the data set. Maybe with data from MusicBrainz. Or other data sets. **The key will be to have a data set large enough.**
- First work with one label per song. Then multi label.
- Add data from MTG Jamendo
- Use beat-sync features: instead of calculating features based on arbitrary time windows, we can align them with beats.
- Use harmonic vs percussive components. Called HPSS: Harmonic-Percussive Source Separation. Then it's possible to extract features separately from each rather than on the original audio.
- Use Mel spectrograms (frequency vs time images) and train CNN models on those.


## To do
- same with sqare root sample weight (later)
- Compare score for each genre between FMA_small and FMA_medium (in both case for class_weight)
- Try BalanceRandomForestClassifier or EasyEnsembleClassifier (designed for imbalanced datasets)
- try XGBoost
- perform ablation on FMA medium and check which class is helping each genre. So that I can maybe complete the feature set to help the genres which have a low F1.
- Is it worth trying another dataset? Maybe one without audio so there is more dataset
- Maybe try genres and not top_genre? And try to predict several genres
- Yes try multilabel, but before, exploratory analysis of genres: how many distinct genres, how many tracks have one, two, three and more genres, how frequently is each genre, which genres frequently occur together, how many genres are sufficiently represented to be useful for training?
This will help understanding the confusion matrix obtained on FMA_medium.


# Road map
## I) Framing the problem and big picture
### Define the objective
Classify songs based on their musical genres and eventually sort the songs and generate playlists based on the genres.
Eventualy, playlist could be generated using mood or emotions.
### How the solution is used
Preliminary. A python code is run on a music library, sets the tag for musical genre on all the files. An alternative tool is used to generate playlists.
Ultimately. Online tool where people can upload their library, get their songs classified and organised into an easy and handy architecture. The online tool will include options to generate playlists, based on requirements. Eventually, it will generate playlist based on mood or emotions.
### Current solutions
- https://picard.musicbrainz.org/
MusicBrainz Picard is a free, open-source music tagging application that identifies, organizes, and updates metadata for digital audio files. It is the official tagger for the MusicBrainz music database and is available for Windows, macOS, and Linux. Its combination of community-maintained metadata and acoustic fingerprinting makes it one of the most powerful tools for cleaning and organizing large music collections. 
- https://essentia.upf.edu/
Essentia is an open-source software library for audio analysis and music information retrieval (MIR). Originally developed by the Music Technology Group at Universitat Pompeu Fabra, it has become a widely used toolkit in both academic research and industry for extracting musical features, analyzing recordings, and building music-aware applications. 
### Problem framing
This problem is a multioutput classification problem. It is multiclass and multilabel. 
For now it's going to be an offline supervised model. Maybe it will involve into a online supervised model. 
We are going to use simple models at first such as random forest. And then eventually try more complexe models.
Could be interesting do see how CNN performs.
### Performances
The performances of the classifier are measured using precision and recall.
precision = TP / (TP + FP)
recall = TP / (TP + FN)
Both are combined into F1 score.
We will most likely focus on a good precision rather than a good recall.

## II) Getting the data
We start with a small sample that is FMA_small.
FMA is an open and easily accessible dataset suitable for evaluating several tasks in MIR, a field concerned with browsing, searching, and organizing large music collections. The community's growing interest in feature and end-to-end learning is however restrained by the limited availability of large audio datasets. The FMA aims to overcome this hurdle by providing 917 GiB and 343 days of Creative Commons-licensed audio from 106,574 tracks from 16,341 artists and 14,854 albums, arranged in a hierarchical taxonomy of 161 genres. It provides full-length and high-quality audio, pre-computed features, together with track- and user-level metadata, tags, and free-form text such as biographies. We here describe the dataset and how it was created, propose a train/validation/test split and three subsets, discuss some suitable MIR tasks, and evaluate some baselines for genre recognition. Code, data, and usage examples are available at https://github.com/mdeff/fma.

FMA_small is: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB).

## III) Exploring the data and first insights
The data is memory heavy. I would probably need to work on some online CPUs.
I can't train model directly on the audio files. There are way too many data to do so. I need to reduce the data. This is done by computing the features of all the audios. Then I can train the models on these features.


## IV) Preparing the data
The key is to compute features for all audios. And choose which features to use, which one to take mean, std etc.
Maybe need to standardize the features.
See the "Feature extraction" section in that document.

## V) Model exploration
Train models and choose the one that performs the best on base features: MFCC + spectral centroid + bandwidth + rolloff + contrast + ZCR + RMS.



## VI) Fine-tuning models

## VII) Final solution

## VIII) Launching and releasing the solution

# Feature extraction
If I'm correct, for each feature, librosa computes the feature on windows of 2048 samples spaced by hops of 512 samples (to confirm). So it gives an array of roughly 517 values.
Then, FMA feature analysis computes the mean, std, skew, kurtosis, median, minimum and maximum. 
It would be interesting to add to that the 10th percentil and the 90 percentil.	

I wrote and ran the code do_features.py that compute the audio features on the FMA dataset. Here is the list of the features I am computing in this code:
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        delta_chroma_stft=12,
        delta_chroma_cqt=12,
        tonnetz_cqt=6,
        tonnetz_cens=6,
        mfcc=20,
        delta_mfcc=20,
        delta2_mfcc=20,
        rms=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_flatness=1,
        spectral_flux=1,
        spectral_rolloff_50=1,
        spectral_rolloff_85=1,
        spectral_rolloff_95=1,
        harmonic_rms=1,
        percussive_rms=1,
        harmonic_ratio=1,
        percussive_ratio=1,
        onset_strength=1,
        tempogram_mean=1,
        beat_interval=1,
All these features are statistical features. It means that for each of those, these statistics are computed: 'mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max', '10th', '90th'.
In addition, there are 2 scalar features: 'tempo' and 'beat_rate'.


# Feature analysis
## List of all features to investigate
### Spectral centroid 
It's basicaly the center of gravity of the frequencies.
Frequency based.
Computationally cheap.

### Spectral bandwidth
It basicaly tells how spread out the frequencies are around the centroid.
Frequency based.
Computationally cheap.

### Spectral rolloff
It is the frequency below which a specified percentage of the energy lies.
It would be good to compute 3 different rolloff frequencies with 3 different percentages: 50% (half), 85% (default), 95% (high frequency tail).
It would be interesting to compute rolloff_95 - rolloff_50 to see how broad the upper-frequency distribution is.	
Frequency based.
Computationally cheap.
This is an interesting feature.

### Spectral contrast
It attempts to characterize the difference between peaks and valleys of energy across frequency bands.
Librosa's default is roughly 7 frequency bands, starting from a minimum frequency and expanding upward approximately by octaves.
Frequency based.
Computationally cheap/moderate.
This one is very interesting.

### Spectral flatness
It measures how noise-like versus tone-like the spectrum is.
Frequency based.
Computationally cheap.
Less interesting but good to keep I guess.

### Spectral flux / spectral change
It tells how quickly is the spectrum changing.
Librosa doesn't have a function for that. But possible to compute from successive STFT frames.
Frequency based.

### Zero-crossing rate
It measures how frequently the waveform crosses zero.
Time/amplitude based.
Computationally cheap.

### RMS energy
It is a measure of the signal's average energy/loudness over a short period.
Nice to have mean, std, max.
Time/amplitude based.
Computationally cheap.

### MFCCs
MFCC = Mel-Frequency Cepstral Coefficients.
They're widely used in speech and audio classification.
The underlying idea is complicated, but the intuition is relatively simple:
MFCCs provide a compact representation of the spectral shape/timbre of an audio signal, using a frequency scale designed to roughly reflect human hearing.
The process is roughly: 
Audio
  ↓
STFT
  ↓
Power spectrum
  ↓
Mel filter bank
  ↓
log
  ↓
DCT
  ↓
MFCCs

MFCCs capture characteristics of timbre.
Frequency/timbre based.
Computationally moderate.
MFCCs would probably be the most important feature family.


### Delta and delta-delta MFCCs
It is basicaly how MFCCs are changing (delta) and how this delta is changing (delta-delta). So basicaly the derivative and second derivative of MFCCs.
Frequency/timbre based.
Computationally moderate.
Also important.

### Chroma
It captures information about harmonic content.
Frequency/pitch based.
Computationally moderate.
This is important.

### Tonnetz 
A more specialized harmonic representation.
Related to chroma but encodes harmonic relationships differently.
Frequency/harmony based.
Computationally moderate.
It is an advanced feature but can be interesting.

### Tempo
Beats per second of the track.
Time/rhythm based.

### Beat / rhythm features
We can also look at:
- beat intervals
- onset density
- rhythmic regularity
- beat strength
- rhythmic patterns
The most important ones (or the first to check) would be tempo, beat positions and onset strength.
From beat positions, we can look at the regularity of beat intervals for instance (with std  of beat intervals for instance).
For onset strenght, look at mean and std at first.
Time/rhythm based.


### Onset rate
It is essentially the beginning of a musical event. A musical event is a sound event, so it can be a note, a drum hit, a vocal syllabe etc.
So the onset rate is how frequently the audio contains detected new musical events.
To use with caution though, because the detection parameters of these events will strongly affect the result.
Time/rhythm based.
It is useful for differenciate rhythmically dense genres.

### Loudness and dynamics
Worth investigating:
- peak amplitude
- crest factor
- dynamic range
- loudness
- loudness variation
Time based.
These features could be interesting for the analysis, although beware:
Production/mixing/mastering can become a shortcut for genre.

### Spectral statistics
For instances:
- spectral skewness: measures the asymmetry of the frequency distribution.
- spectral kurtosis: measures how concentrated or extreme the distribution is.
Frequency based.
Maybe not as important.


## Feature families
Based on the features I computed with do_features.py
- Timbre: MFCC, delta_MFCC, delta2_MFCC
- Spectral shape: centroid, bandwidth, rolloff, contrast, flatness, flux
- Harmony / pitch: chroma_stft, chroma_cqt, chroma_cens, delta_chroma_\*, tonnetz_\*
- Rhythm: tempo, beat_rate, beat_interval, tempogram_mean, onset_strength
- Energy / dynamics: RMS, ZCR, harmonic_rms, percussive_rms, harmonic_ratio, percussive_ratio

See how each family affects the performances of the classifier.


## Feature correlation
### Correlation between families
Mean |r| between families (diagonal = within family):
                     chroma  tonnetz  mfcc  spectral_shape  energy_dynamics  harmonic_percussive  rhythm
chroma                 0.14     0.11  0.12            0.14             0.12                 0.16    0.09
tonnetz                0.11     0.18  0.07            0.11             0.08                 0.12    0.06
mfcc                   0.12     0.07  0.20            0.16             0.16                 0.18    0.16
spectral_shape         0.14     0.11  0.16            0.24             0.22                 0.24    0.13
energy_dynamics        0.12     0.08  0.16            0.22             0.29                 0.28    0.13
harmonic_percussive    0.16     0.12  0.18            0.24             0.28                 0.37    0.14
rhythm                 0.09     0.06  0.16            0.13             0.13                 0.14    0.29

This means that overall the famiilies are not much correlated with eachothers.

Family	Spearman mean \(|r|\) within family
Harmonic/percussive	0.37
Energy/dynamics	0.29
Rhythm	0.29
Spectral shape	0.24
MFCC	0.20
Tonnetz	0.18
Chroma	0.14

with those families:
  chroma:               [chroma_stft, chroma_cqt, chroma_cens, delta_chroma_stft, delta_chroma_cqt]
  tonnetz:              [tonnetz_cqt, tonnetz_cens]
  mfcc:                 [mfcc, delta_mfcc, delta2_mfcc]
  spectral_shape:       [spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness,
                        spectral_flux, spectral_rolloff_50, spectral_rolloff_85, spectral_rolloff_95]
  energy_dynamics:      [rms, zcr]
  harmonic_percussive:  [harmonic_rms, percussive_rms, harmonic_ratio, percussive_ratio]
  rhythm:               [onset_strength, tempogram_mean, beat_interval, tempo, beat_rate]

### Strong correlation between features
- harmonic_rms / rms: mean, median, min, max, 10th and 90th: 0.95+
- harmonic_ratio / percussive_ratio: mean, median, std, 10th, 90th, skew, kurtosis: 0.99+ (min and max: 0.91+)
- tonnetz_cqt / tonnetz_cens: mean, median, 10th, 90th: 0.90+ for all 6 tonnetz
- beat_interval / tempo: mean, median, 10th, 90th: 0.94+
- beat_interval / beat_rate: mean, median: 0.91+
- tempo / beat_rate : 0.91

### Clusters and suggests of removal (pruning)
- beat_rate and tempo: 0 % kept
- percussive_ratio: 11.1 % kept
- spectral_flux: 22.2 % kept
- zcr, rms, tempogram_mean, spectral_flatness: 33.3 % kept
- percussive_rms: 44.4 % kept

  ### Redundancy
from highest to lowest
- percussive/harmonic ratio
- tempogram_mean
- rms
- harmonic_rms
- tempo
...
  See more in feature_redundancy.png



# Investigating FMA small
The idea is to use FMA small to get a first guess of which models perform ok on the dataset, and which features are useful.
## Roadmap
- On the FMA_small dataset and with the full feature set, choose 3 models that performs the best.
- check with different seeds for a better measurement of F1 scores.
- Then perform ablation. Check the most computation expensive features. See if we can remove them from the feature computation without degrading to much the performances.
- Check all family features with ablation.
- Check statistics with ablation as well.

## Chosen models
These models are the best on CV folds:
- svm : scaler=quantile, gamma=0.000537 (m=0.75), C=1.7 (run_15) ; fit time = 130
- svm: scaler=standard, gamma=0.000537 (m=0.75), C=1.3 (run_best_svm) ; fit time = 53
- mlp : scaler=standard, alpha=6, hidden_layer_sizes=(64,) (run_19) ; fit time = 30
- hgb : l2_regularization=0, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=10 (run_22) ; fit time = 600

Now check with 4 seeds and on test fold:
[seed 2 | full    | 1397 feats] svm    macro-F1=0.556±0.021  acc=0.560±0.020  (1035s, 16 configs x 5 folds)  C=2.2, gamma=0.000537
[seed 2 | full    | 1397 feats] hgb    macro-F1=0.561±0.018  acc=0.565±0.015  (2258s, 12 configs x 5 folds)  l2_regularization=1, learning_rate=0.1, max_leaf_nodes=31
[seed 2 | full    | 1397 feats] mlp    macro-F1=0.557±0.012  acc=0.564±0.013  (898s, 35 configs x 5 folds)  alpha=10, hidden_layer_sizes=(64, 32)
[seed 3 | full    | 1397 feats] svm    macro-F1=0.557±0.019  acc=0.563±0.018  (1031s, 16 configs x 5 folds)  C=1.3, gamma=0.000537
[seed 3 | full    | 1397 feats] hgb    macro-F1=0.564±0.016  acc=0.568±0.017  (2249s, 12 configs x 5 folds)  l2_regularization=1, learning_rate=0.1, max_leaf_nodes=17
[seed 3 | full    | 1397 feats] mlp    macro-F1=0.556±0.020  acc=0.564±0.022  (926s, 35 configs x 5 folds)  alpha=10, hidden_layer_sizes=(128, 64)
[seed 4 | full    | 1397 feats] svm    macro-F1=0.549±0.012  acc=0.553±0.010  (1029s, 16 configs x 5 folds)  C=1.3, gamma=0.000537
[seed 4 | full    | 1397 feats] hgb    macro-F1=0.562±0.009  acc=0.566±0.007  (2255s, 12 configs x 5 folds)  l2_regularization=0, learning_rate=0.05, max_leaf_nodes=31
[seed 4 | full    | 1397 feats] mlp    macro-F1=0.548±0.018  acc=0.554±0.014  (884s, 35 configs x 5 folds)  alpha=10, hidden_layer_sizes=(128, 64)
[seed 5 | full    | 1397 feats] svm    macro-F1=0.555±0.006  acc=0.559±0.005  (1030s, 16 configs x 5 folds)  C=1.3, gamma=0.000537
[seed 5 | full    | 1397 feats] hgb    macro-F1=0.561±0.014  acc=0.563±0.014  (2250s, 12 configs x 5 folds)  l2_regularization=0, learning_rate=0.1, max_leaf_nodes=31
[seed 5 | full    | 1397 feats] mlp    macro-F1=0.548±0.011  acc=0.556±0.010  (949s, 35 configs x 5 folds)  alpha=10, hidden_layer_sizes=(128, 64)

=== Best configuration per model (cross-validated; f1_worst = worst config of that model) ===
feature_set  n_features model  f1_mean  f1_std  f1_seed_std  f1_worst  acc_mean  n_configs  fit_time_s  n_iter                                                    config
       full        1397   hgb    0.561   0.013        0.001     0.184     0.565         12     371.120 100.000 l2_regularization=1, learning_rate=0.1, max_leaf_nodes=31
       full        1397   svm    0.554   0.015        0.003     0.543     0.559         16     124.571     NaN                                     C=1.3, gamma=0.000537
       full        1397   mlp    0.551   0.017        0.005     0.460     0.558         35      45.397 104.400                     alpha=10, hidden_layer_sizes=(64, 32)

TEST  full/mlp: acc=0.500  macro-F1=0.494
TEST  full/hgb: acc=0.495  macro-F1=0.490
TEST  full/svm: acc=0.477  macro-F1=0.476


Difference between CV and test set is big. Have to investigate.

Here is genre densities accross train and test set:
Genre Hip-Hop: 12.50% in train and 12.50% in test
Genre Pop: 12.50% in train and 12.50% in test
Genre nan: 0.00% in train and 0.00% in test
Genre Rock: 12.50% in train and 12.50% in test
Genre Experimental: 12.50% in train and 12.50% in test
Genre Folk: 12.50% in train and 12.50% in test
Genre Jazz: 0.00% in train and 0.00% in test
Genre Electronic: 12.50% in train and 12.50% in test
Genre Spoken: 0.00% in train and 0.00% in test
Genre International: 12.50% in train and 12.50% in test
Genre Soul-RnB: 0.00% in train and 0.00% in test
Genre Blues: 0.00% in train and 0.00% in test
Genre Country: 0.00% in train and 0.00% in test
Genre Classical: 0.00% in train and 0.00% in test
Genre Old-Time / Historic: 0.00% in train and 0.00% in test
Genre Instrumental: 12.50% in train and 12.50% in test
Genre Easy Listening: 0.00% in train and 0.00% in test

So nothing wrong here.
Worth noticing that the genre are evenly distributed in the small test.
Evenly distributed.
The F1 score drop for Folk genre. And folk genre in test set is dominated at 91% by the same artist. So choose best models on CV and move on:


## Ablation
From correlation analysis, here is the suggested list of features for ablation: 
 - harmonic / percussive
 - rms
 - tonnetz_cqt / tonnetz_cens
 - beat_interval, tempo, beat_rate
 - tempogram_mean
 - spectral_flatness
Some of them are higly correlated some perform ablation by steps.

### First results
MLP seems more affected by ablation than HGB.
First results suggest that:
- harmonic_percussive is safe to remove:
	delta_F1 = 0.002 for HGB and 0.005 for MLP (check SVM)
- MFCC, Rhythm features,  are important 
- Tonnetz and Energy dynamics are promising candidates
- Chroma and maybe all CQT based features potential candidates for HGB but not for MLP
- Tonnetz_cqt most important one, tonnetz_cens doesn't contribute as much (especially for HGB)

### First conclusions
I ran the ablation and removed in order the most computationaly expensive features. Here are the results:
- SVM:
config  cols_removed  n_remaining  score_mean  score_std  delta_vs_full_mean  delta_vs_full_std  p_vs_full
  full             0         1397      0.5533     0.0212              0.0000                NaN        NaN
  no_1            36         1361      0.5532     0.0221             -0.0001             0.0033     0.9414
  no_2           468          929      0.5454     0.0180             -0.0079             0.0099     0.1793
  no_3           479          918      0.5396     0.0154             -0.0137             0.0104     0.0339
- HGB:
config  cols_removed  n_remaining  score_mean  score_std  delta_vs_full_mean  delta_vs_full_std  p_vs_full
  full             0         1397      0.5598     0.0210              0.0000                NaN        NaN
  no_1            36         1361      0.5579     0.0191             -0.0019             0.0069     0.6304
  no_2           468          929      0.5548     0.0182             -0.0050             0.0108     0.4253
  no_3           479          918      0.5478     0.0168             -0.0121             0.0102     0.0546
- MLP:
--- configurations vs the full set | f1_macro (delta < 0 = worse; p-values approximate) ---
config  cols_removed  n_remaining  score_mean  score_std  delta_vs_full_mean  delta_vs_full_std  p_vs_full
  full             0         1397      0.5522     0.0205              0.0000                NaN        NaN
  no_1            36         1361      0.5474     0.0215             -0.0048             0.0160     0.6031
  no_2           468          929      0.5408     0.0170             -0.0114             0.0128     0.1357
  no_3           479          918      0.5355     0.0162             -0.0167             0.0113     0.0197

no_1 is full - HP ; no_2 is full - HP - cqt ; no_3 is full - HP - cqt - beat_tracking

First of all: MLP is very sensitive to ablation. SVM is insensitive to ablation of HP. HGB is a little bit.
So HP is safe to remove. Even for MLP, the loss is not that big.
Concerning cqt, it can be considered to be removed. Especially for HGB and SVM.
HP is 69% of the compute time. But once removed, cqt represents 24% of the remaining compute time (taking into account all features, so chroma cqt and cens, their delta and their tonnetz).
For now I will keep it. And check those results again with more data on the medium dataset, before taking a decision for the bigger datasets.


# Investigating FMA medium
## Features extraction
Done with do_features.
I forgot to remove HP. So it's in the features csv file. It will allow me to do the ablation again with much more data.
But first I need to retrain the models.

## Model comparison
The first model comparison gives very degraded performances compared to FMA_small. Around 0.35 macro-F1 score on average.

## Investigation of difference of performances
- I checked the features.csv files. Nothing seems wrong here. There is no excess of NaN or Inf values:
FMA_small
Shape: (8000, 1397)
Infinite values: 1
NaN values: 8915
Tracks with NaN or Inf values: 6
and
FMA_medium
Shape: (25000, 1397)
Infinite values: 1
NaN values: 29588
Tracks with NaN or Inf values: 20

- I checked that using myfeatures_medium.csv but restrained on the small dataset gives exactly the same results (tested with mlp, results in mlp_only_small/ in fma_medium/ results).
I also checked that using myfeatures_medium.csv but restrained to the FMA_small genres. The performances are very degraded as well. With the same model (best mlp on FMA_small) that gives 0.550 macro-F1 on FMA_small, I get 0.424 macro-F1.

There is an issue with genre having 0 F1 score. Because apparently the number of tracks for this genre is very low.
There is a mismatch between number of tracks in FMA_medium (and entries in the features medium csv) and the number of tracks used in compare_models.py.
For Blues, Classical, Country, Easy Listening (low number of tracks), all the tracks are used in compare_models.py
For Elctronic, only 4780 over 5682 tracks are used. 
Experimental: 1126 over 2026
Folk: 466 over 1367
Hip hop: 1075 over 1981
Instrumental: 275 over 1176
International: 16 over 916
Jazz: all
Old-Time / Historic: 459 all
Pop: 167 over 1067
Rock: 5489 over 6392
Soul-RnB: all
Spoken: all 

I found out the issue. I was having a bug in compare_models.py in the model selection. I was selecting only medium, and not medium + small. So I was missing 7200 tracks in the training.


I did some investigation. I found a bug in the code that do the model comparison. I was not using the full FMA medium dataset. I was using only 17,000 over the 25,000 tracks. And some genres were very poorly populated.

Now I train my model on the full FMA medium dataset. But the performances are still low (macro-F1 barely reach 0.40 with best SVM model, it was 0.55 on FMA_small).

I also investigated few other things:

- when I restrict the FMA_medium dataset to only the tracks which are in the FMA_small dataset, and I train the MLP model with the best parameters obtained in my previous analysis (on FMA_small), I obtain exactly the same results as before.
- when I restrict the FMA_medium dataset to only the tracks whose genre is in the FMA_small genre (8 genres out of the 16 genres in FMA_medium), and I train the same MLP model (with best parameters), I get way worse performances: macro-F1 drops from 0.55 to 0.42
- I checked that the feature extraction is identical in both cases: same 1,397 feature columns, same column names and ordering, no duplicate or missing column.
- I checked the missing and NaN values. Nothing seems wrong here:

FMA_small  
Shape: (8000, 1397)  
Infinite values: 1  
NaN values: 8915  
Tracks with NaN or Inf values: 6  
<br/>FMA_medium  
Shape: (25000, 1397)  
Infinite values: 1  
NaN values: 29588  
Tracks with NaN or Inf values: 20

- I have been trying different models (SVM, RF, ET, HGB, MLP) with different hyperparameters. They all experience performance drop from FMA_small to medium.

Here is the MLP model performances with training restricted to tracks from FMA_small:
feature_set  n_features model  f1_m_mean  f1_m_std  f1_m_worst  f1_w_mean  n_configs  fit_time_s  n_iter                            config
       full        1397   mlp      0.550     0.016       0.550      0.551          1      10.329  84.600 alpha=6, hidden_layer_sizes=(64,)
\-\-- full / mlp (seed 0) (out-of-fold predictions) ---  
precision recall f1-score support  
<br/>Electronic 0.566 0.561 0.563 899  
Experimental 0.474 0.466 0.470 899  
Folk 0.624 0.614 0.619 900  
Hip-Hop 0.678 0.713 0.695 897  
Instrumental 0.530 0.526 0.528 900  
International 0.623 0.624 0.624 900  
Pop 0.307 0.288 0.297 900  
Rock 0.603 0.638 0.620 899  
<br/>accuracy 0.554 7194  
macro avg 0.551 0.554 0.552 7194  
weighted avg 0.551 0.554 0.552 7194

And here is the MLP model performances with training restricted to tracks whose genre is among the FMA_small genres:
feature_set  n_features model  f1_m_mean  f1_m_std  f1_m_worst  f1_w_mean  n_configs  fit_time_s  n_iter                            config
       full        1397   mlp      0.522     0.019       0.522      0.649          1      40.970 115.600 alpha=6, hidden_layer_sizes=(64,)
\-\-- full / mlp (seed 0) (out-of-fold predictions) ---  
precision recall f1-score support  
<br/>Electronic 0.680 0.798 0.734 5679  
Experimental 0.532 0.440 0.482 2025  
Folk 0.592 0.607 0.599 1366  
Hip-Hop 0.723 0.681 0.701 1972  
Instrumental 0.353 0.286 0.316 1175  
International 0.590 0.402 0.478 916  
Pop 0.258 0.037 0.065 1067  
Rock 0.756 0.859 0.804 6388  
<br/>accuracy 0.672 20588  
macro avg 0.560 0.514 0.523 20588  
weighted avg 0.643 0.672 0.649 20588

**Conclusion: the performances are not too bad when restricting to the 8 genres from FMA_small. They are the most populated genres. So adding less populated genres adds noise and degrades the performances of the classifier. It is a much harder classification task with the 16 genres.**

- Proportion of genres
Genre	Number of tracks	Proportion
Electronic	5,679	27.6%
Experimental	2,025	9.8%
Folk	1,366	6.6%
Hip-Hop	1,972	9.6%
Instrumental	1,175	5.7%
International	916	4.4%
Pop	1,067	5.2%
Rock	6,388	31.0%
Total	20,588	100%

- Comparison of F1 score from FMA_small to FMA_medium
Genre	FMA_small F1	FMA_medium F1
Electronic	0.563	0.734
Experimental	0.470	0.482
Folk	0.619	0.599
Hip-Hop	0.695	0.701
Instrumental	0.528	0.316
International	0.624	0.478
Pop	0.297	0.065
Rock	0.620	0.804

The model does way better on Rock and Electronic because they represent each 31% and 28% of the dataset (it was 12.5% on FMA_small). And it performs same or worst on the other genres.
Pop in particular drops severely. Because it is less populated (5%), the model is likely biased towards more populated genres (Rock). Indeed, in the confusion matrix, Pop tracks are predicted as Rock in 42% and Electronic in 27%.

--> Possibility of using class_weight = 'balanced' to balance the sample according to class density. Using this with the best MLP from FMA_small, I go from
macro-F1 = 0.362 --- weighted-F1 = 0.619
to
macro-F1 = 0.389 --- weighted-F1 = 0.583
Maybe the effect will be bigger with a better model?
Anyway, its a trade of between most populated classes and least populated classes.

### Best models so far
Using all the training set improves a bit the performances (training set more balanced).
Using class_weight = 'balanced' also improves a bit the performances.
I'm gonna try undersampling and oversampling to see if it also improves performances. 

For now, here are the best models I got:
- SVM balanced (05_svm_balanced): [seed 0 | full    | 1397 feats] svm    macro-F1=0.435±0.023  weighted-F1=0.637±0.020  (21213s, 20 configs x 5 folds)  C=2.5, gamma=0.000358
- SVM balanced quantile (05_svm_balanced_quantile): [seed 0 | full    | 1397 feats] svm    macro-F1=0.436±0.022  weighted-F1=0.628±0.021  (24124s, 20 configs x 5 folds)  C=1, gamma=0.000358
Could be worth investigating lower values for C.
- HGB balanced (05_hgb_balanced): [seed 0 | full    | 1397 feats] hgb    macro-F1=0.410±0.014  weighted-F1=0.607±0.019  (13471s, 12 configs x 5 folds)  class_weight=balanced, l2_regularization=1, learning_rate=0.1, max_leaf_nodes=5, min_samples_leaf=1
- MLP balanced (05_mlp_balanced): [seed 0 | full    | 1397 feats] mlp    macro-F1=0.404±0.016  weighted-F1=0.603±0.020  (704s, 12 configs x 5 folds)  alpha=0.3, class_weight=balanced, hidden_layer_sizes=(256, 128)

and the non balanced models:
- SVM (04_svm): [seed 0 | full    | 1397 feats] svm    macro-F1=0.408±0.011  weighted-F1=0.631±0.015  (36464s, 45 configs x 5 folds)  C=8, gamma=0.000358
- SVM quantile (04_svm_quantile): [seed 0 | full    | 1397 feats] svm    macro-F1=0.412±0.019  weighted-F1=0.632±0.016  (25355s, 24 configs x 5 folds)  C=8, gamma=0.000215
- HGB (05_hgb): [seed 0 | full    | 1397 feats] hgb    macro-F1=0.384±0.022  weighted-F1=0.622±0.017  (8068s, 12 configs x 5 folds)  class_weight=None, l2_regularization=1, learning_rate=0.1, max_leaf_nodes=11, min_samples_leaf=1
- MLP (05_mlp): [seed 0 | full    | 1397 feats] mlp    macro-F1=0.388±0.017  weighted-F1=0.611±0.016  (1245s, 12 configs x 5 folds)  alpha=0.3, class_weight=None, hidden_layer_sizes=(256,)
- 


For now SVM is performing the best.


### Undersampling
I tried using undersampling for best SVM, HGB and MLP models described above.
I tried for undersampling = 500 and undersampling = 2000 (so 400 and 1600 of each genre in each fold). I trained the best normal and the best balanced model in both cases.
In all cases, the models perform better with the whole dataset and class_weight = 'balanced'.
So undersampling is not helping.

### Oversampling
- oversampling 200: very tiny gain (0.001) for SVM and HGB but loss for MLP. I guess MLP is overfitting. It is performing systematicaly worst on the genres that have been oversampled. So maybe try to train MLP model with a stronger regularisation, or less depth.
- oversampling 1000: SVM tiny gain (0.002), HGB small loss (0.001), MLP big loss (0.02)

I tried training the models again while oversampling. Here are the results:
- SVM: small gain: 0.001 on macro F1 and 0.006 on weighted F1. Best model is oversampling_1000 with SVM quantile balanced and C=1 and gamma=0.000358. (results in oversampling_1000_svm_quantile_balanced_02, compared with 05_svm_balanced_quantile)
- HGB: small gain 0.004 on macro F1 and 0.003 on weighted F1. Best model is oversampling 1000
with HGB balanced and learning_rate=0.1, l2=3, max_leaf_nodes=5 and min_samples_leaf=1 (results in oversampling_1000_hgb_balanced_02 compared to 05_hgb_balanced)
- MLP: 

## Conclusions
Best model is:
- SVM, with oversampling=1000, scaler=quantile, class_weight=balanced, C=1 and gamma=0.000358 (oversampling_1000_svm_quantile_balanced_02)
- or same without oversampling (05_svm_balanced_quantile)

The macro F1 score is 0.43 - 0.44 and weighted F1 is about 0.63.

Possible things to do to try to improve performances:
- try with sqare root sample weights
- Try BalanceRandomForestClassifier or EasyEnsembleClassifier (designed for imbalanced datasets)
- try XGBoost
- perform ablation on FMA medium and check which feature family is helping each genre. So that I can maybe complete the feature set to help the genres which have a low F1 score.
- Try to improve or complete the dataset (especially the lowest populated genres). Try another dataset.





# Chatbot prompts

Hello. I am working on a data science project. I am trying to classify music tracks by their genre. I am working with the FMA dataset.
I already wrote a code that computes audio features from the audio tracks. I computed these "statistical" features:
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        delta_chroma_stft=12,
        delta_chroma_cqt=12,
        tonnetz_cqt=6,
        tonnetz_cens=6,
        mfcc=20,
        delta_mfcc=20,
        delta2_mfcc=20,
        rms=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_flatness=1,
        spectral_flux=1,
        spectral_rolloff_50=1,
        spectral_rolloff_85=1,
        spectral_rolloff_95=1,
        harmonic_rms=1,
        percussive_rms=1,
        harmonic_ratio=1,
        percussive_ratio=1,
        onset_strength=1,
        tempogram_mean=1,
        beat_interval=1,
each of these features comes with 9 statistics: mean, std, median, min, max, skew, kurtosis, 10th and 90th percentiles. 
In addition, I have 2 scalar features: tempo and beat_rate.
In total it makes 1397 features.

I first started with FMA_small. I computed the features for this dataset. Then I compared different classes of models. I selected 3 promisins models: SVM, HGB and MLP. I got around 0.55 macro-F1 score for the best models.

Now I am going to FMA_medium. I computed the features for this dataset. And I am comparing the same classes of models. But the performances are way worst. I barely reach 0.40 macro-F1 on this dataset with the best models.


# Reports and time spent
### 15/09/2026
Approx 5 hours.
Creating of GitHub repo. Retrieving of data. First look at data

### 16/09/2026
Approx 4 hours.
Understanding how to load the audio files. Spend some time to configure emacs correctly. Checked what docstrings to use.

### 17/09/2026
Approx 5 hours.
I've been taking some time to properly read the data. I've spent time to read and study about feature engineering, what features are important, how to compute them etc.

### 18/09/2026
Approx 4 hours.
Spend some time to understand the code features.py from FMA. Especially the multiprocessing part. I implemented their code into the project, and tried to adapt it. There are still some bug to fix.

### 22/09/2026
Approx 4 hours.
Finally debuged the function feature_stats. It was linked to differences in panda versions. The FMA code was using very old version of pandas that was much more permisive. The code `do_features.py` is running now. But a lot of audio files are not read correctly though.

### 23/09/2026
Approx 5 hours.
Fully debuged `do_features.py`. And I added many more features. It's now complete.
I did a version with compute time analysis (`timed_do_features.py`), and run the analysis for 100 audio files. 
I also wrote a version of `do_features.py` which is cleaned, with docstrings and comments.
Next step is to analyse the results of the computational times. And to extract the features from all the files.

### 25/09/2026
Approx 3 hours.
Did some configuration of Emacs for Latex. Started creating a template of .tex file for explaining the compute time analysis results.
I added few missing feature computations to `do_features.py`. And updated `timed_do_features.py` with the new features.
I just need to figure out the chatgpt explanation on tempogram, `update do_features.py` and then update again `timed_do_features,py`

 ### 28/09/2026
 Approx 2 hours.
 I updated `timed_do_features.py` with the new features. I ran again the computational time analysis with 200 audio files. I started writting the results of the analysis in `compute_time_analysis.tex`.

 ### 29/09/2026
 Approx 5 hours.
 I finally finish to write compute_time_analysis.tex. I made some nicer plots to include in the document.
 I also started thinking about the comparison of classifiers.

 ### 30/09/2026
 Approx 5 hours.
 I started writting a code to compare models. I have now two codes `feature_groups.py` and `compare_models.py`. Together they define several sets of features, and a set of models and train these models with different sets of parameters on the different feature sets.

### 05/10/2026
Approx 3 hours.
I ran `compare_models.py` a first time, on the compact set and the full set, for 6 different models. I have been investigating the results. They are stored in results/run_01. I have to run several more runs to investigate more the grid parameters (most of the best models were on the edge of the first grid).

### 06/10/2026
Approx 3 hours.
I did some modifications on the code `compare_models.py`. Now models are ranked according to F1 scores. I ran `compare_models.py` two more times, with different sets of parameters. I am narrowing down the models.
I also tried hgb model with different max_iter, from 50 to 500. And there is an increase in F1 score up to 300. Then from 300 to 500 there is no such increase in F1 score. And the fitting time is more or less the same. (run_04)
I tried it on the full set. F1 score stops increasing starting at max_iter=200. The results are actually already good after 50 iterations. Half a point on F1 from 50 to 100 and half a point from 100 to 200. Then more or less constant up to 1000. (run_05)

### 07 and 08/10/2026
Approx 3x5 hours
I continued comparing models. I chose 3 different best models, one SVM, one HGB and one MLP. I just need to measure the performances of those models with different seeds and check on the test set. (The code didn't run last night.)
I also started writting a code for ablation and one for permutation. I want to investigate feature importance. Especially the compationnaly expensive features. I also wrote a code to check on feature correlations. To try to reduce the feature set.

### 12/10/2026
3 hours.
I investigated the correlations of the features.
I also checked the models on the test set. There is a significant difference of performances between CV out of fold and test set. I investigated what could be the cause of this difference.

### 14 and 15/10/2026
3 and 4 hours.
I continued investigating the difference in performances. I started writing a report to explain the analysis and gather the results.

### 19/10/2026
5 hours.
I finished writing the report to explain the first model comparison. I also understood why there is such a difference in F1 measured on the CV and on test set. Because Folk genre in test set is dominated at 91% by the same artist.

### 20/10/2026
5 hours.
I started investigating correlations. I now have a list of features to investigate with ablation. I started doing the first runs of the ablation process. With the best SVM, HGB and MLP.

### 22/10/2026
6 hours.
I annalyzed the first runs of ablation. Harmonic_percussive does not contribute a lot, so it is safe to get rid of it. Cqt contributes a bit more, but still represents 25% of the compute time once HP has been removed. So maybe worth getting rid of.

### 27/10/2026
A started investigating the FMA medium dataset. I did the first runs of compare_models.py.

### 28/10/2026
There is a very big drop in performances between FMA small and FMA medium. Maybe because there are 16 genres instead of 8. But there is also an issue with the number of tracks. I got like 15,000 instead of 25,000, so some genre are very poorly represented. I have to figure this out.

### 29/10/2026
I found a bug in compare_models.py. The selection of the subset was wrong and was selecting only the medium subset, not medium + small. I started new run of compare_models.py with the correct model selection (runs are named model_03 and on going).

### 30/10/2026
I investigated the drop in performances. I did some test with a MLP model. My guess is that the imbalance of class + having 16 genres instead of 8 make the task way more difficult.

### 02/11/2026
5 hours.
I finished investigating best models for FMA_medium and I started runing oversampling and undersampling analysis

### 03/11/2026
6 hours.
I ran several undersampling and oversampling analysis. I also wrote a document explaining the correlation analysis. And started writing the document which presents the ablation analysis.

### 04/11/2026
4 hours.
I continued writing the ablation document. I also started looking for what's next: more data, or another dataset, or going for multilabel classification.
