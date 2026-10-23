# This code is highly inspired by:
# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.
# and has been modified by Thomas Etourneau, thomas.etourneau@tutanota.com

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [Essentia](http://essentia.upf.edu) (C++ with Python bindings)
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)


import os
import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
import utils
from pathlib import Path


# Paths
DATA_DIR = Path(os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/Database'))
AUDIO_DIR = DATA_DIR / "fma_medium"
TRACKS_PATH = DATA_DIR / 'fma_metadata' / 'tracks.csv'
FEATURES_PATH = DATA_DIR / 'fma_metadata' / 'myfeatures_medium.csv'
FAILED_TIDS_PATH = DATA_DIR / 'fma_metadata' / 'failed_tids_medium.npy'

# Used for test and debug
DEBUG_LIMIT = None  # max number of files to load. Set to None if no use
DEBUG_SUBSET = 'medium'


def save(features: pd.DataFrame, ndigits: int) -> None:
    """Sort and write the feature table to disk as a CSV file.

    Args:
        features: Table of extracted features, indexed by track id, with a
            ``MultiIndex`` of columns as produced by :func:`columns`.
        ndigits: Number of significant digits to keep when formatting
            floats in the output CSV.
    """
    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv(FEATURES_PATH, float_format='%.{}e'.format(ndigits))


def test(features: pd.DataFrame, ndigits: int) -> None:
    """Sanity-check the saved features against the in-memory table.

    Reports any track with missing (NaN) values, then reloads the CSV
    written by :func:`save` and checks it matches ``features`` within
    tolerance.

    Args:
        features: Table of extracted features to compare against the
            saved CSV.
        ndigits: Number of significant digits used when the features were
            saved; used to derive the comparison tolerance.
    """
    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print('Failed tracks: {}'.format(', '.join(str(i) for i in indices)))

    tmp = utils.load(FEATURES_PATH)
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


def columns() -> pd.MultiIndex:
    """Build the column layout of the feature table.

    Most features get one column per (sub-feature, statistical moment)
    pair, e.g. ``('mfcc', 'mean', '03')``. A few features (``tempo``,
    ``beat_rate``) are plain scalars and only get a single ``'value'``
    column instead of the full set of moments.

    Returns:
        A sorted ``MultiIndex`` with levels ``('feature', 'statistics',
        'number')``, suitable for indexing the feature ``DataFrame``.
    """
    feature_sizes = dict(
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
    )

    scalar_features = ('tempo', 'beat_rate')  # single value, no moments

    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max', '10th', '90th')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i + 1)) for i in range(size))
            columns.extend(it)

    for name in scalar_features:
        columns.append((name, 'value', '01'))

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def _feature_stats(features: pd.Series, name: str, values: np.ndarray) -> None:
    """Compute and store distribution statistics for one feature.

    Args:
        features: Feature vector being filled in by :func:`compute_features`;
            modified in place.
        name: Feature name, matching a key used in :func:`columns`.
        values: Array of shape ``(n_features, n_frames)`` (or
            ``(n_frames,)``, which is promoted to 2D) holding the
            feature's values over time.
    """
    values = np.atleast_2d(values)  # ensure shape (n_features, n_frames)

    stats_map = {
        'mean': np.mean(values, axis=1),
        'std': np.std(values, axis=1),
        'skew': stats.skew(values, axis=1),
        'kurtosis': stats.kurtosis(values, axis=1),
        'median': np.median(values, axis=1),
        'min': np.min(values, axis=1),
        'max': np.max(values, axis=1),
        '10th': np.percentile(values, 10, axis=1),
        '90th': np.percentile(values, 90, axis=1),
    }

    for moment, arr in stats_map.items():
        cols = [(name, moment, '{:02d}'.format(i + 1)) for i in range(len(arr))]
        positions = features.index.get_indexer(cols)
        features.iloc[positions] = arr.astype(np.float32)


def _feature_scalar(features: pd.Series, name: str, value: float) -> None:
    """Store a scalar feature (single value, no distribution stats).

    Args:
        features: Feature vector being filled in by :func:`compute_features`;
            modified in place.
        name: Feature name, matching a key in ``scalar_features`` within
            :func:`columns`.
        value: The scalar value to store.
    """
    col = (name, 'value', '01')
    if col in features.index:
        features.loc[col] = np.float32(value)


def compute_features(tid: int) -> pd.Series:
    """Extract the full audio feature vector for one FMA track.

    Loads the track's audio, computes a range of spectral, chroma, and
    rhythmic features with librosa, and summarizes each one with
    distribution statistics (mean, std, skew, kurtosis, median, min, max,
    10th/90th percentile) via :func:`_feature_stats`. Scalar features
    (tempo, beat rate) are stored as a single value instead, via
    :func:`_feature_scalar`.

    Args:
        tid: Track id, used to locate the audio file and label the
            returned ``Series``.

    Returns:
        A ``Series`` indexed by the ``MultiIndex`` from :func:`columns`,
        named after ``tid``. Entries stay ``NaN`` if extraction failed.
    """
    filepath = utils.get_audio_path(AUDIO_DIR, tid)
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', module='librosa')
            x, sr = librosa.load(filepath, sr=None, mono=True)

        hop_length = 512
        n_fft = 2048

        # ========================================================
        # ZERO CROSSING RATE
        # ========================================================
        zcr = librosa.feature.zero_crossing_rate(x, frame_length=n_fft, hop_length=hop_length)
        _feature_stats(features, 'zcr', zcr)
        del zcr

        # ========================================================
        # CQT
        # ========================================================
        # Computed ONCE and reused for chroma_cqt, chroma_cens,
        # tonnetz_cqt, and tonnetz_cens.
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=hop_length, bins_per_octave=12, n_bins=7 * 12, tuning=None))

        # ========================================================
        # CHROMA CQT + DELTA
        # ========================================================
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, sr=sr, n_chroma=12, n_octaves=7)
        _feature_stats(features, 'chroma_cqt', chroma_cqt)
        delta_chroma_cqt = librosa.feature.delta(chroma_cqt)
        _feature_stats(features, 'delta_chroma_cqt', delta_chroma_cqt)
        del delta_chroma_cqt

        # ========================================================
        # CHROMA CENS
        # ========================================================
        chroma_cens = librosa.feature.chroma_cens(C=cqt, sr=sr, n_chroma=12, n_octaves=7)
        _feature_stats(features, 'chroma_cens', chroma_cens)

        # ========================================================
        # TONNETZ FROM CQT CHROMA
        # ========================================================
        tonnetz_cqt = librosa.feature.tonnetz(chroma=chroma_cqt, sr=sr)
        _feature_stats(features, 'tonnetz_cqt', tonnetz_cqt)

        # ========================================================
        # TONNETZ FROM CENS CHROMA
        # ========================================================
        tonnetz_cens = librosa.feature.tonnetz(chroma=chroma_cens, sr=sr)
        _feature_stats(features, 'tonnetz_cens', tonnetz_cens)

        del cqt  # no longer needed
        del chroma_cqt
        del chroma_cens
        del tonnetz_cqt
        del tonnetz_cens

        # ========================================================
        # STFT
        # ========================================================
        # Computed ONCE and reused for chroma_stft, RMS, spectral
        # centroid/bandwidth/contrast/flatness/rolloff, and the mel
        # spectrogram.
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
        power_stft = stft ** 2

        # ========================================================
        # CHROMA STFT + DELTA
        # ========================================================
        chroma_stft = librosa.feature.chroma_stft(S=power_stft, sr=sr, n_fft=n_fft, hop_length=hop_length)
        _feature_stats(features, 'chroma_stft', chroma_stft)
        delta_chroma_stft = librosa.feature.delta(chroma_stft)
        _feature_stats(features, 'delta_chroma_stft', delta_chroma_stft)
        del chroma_stft
        del delta_chroma_stft

        # ========================================================
        # RMS
        # ========================================================
        rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop_length)
        _feature_stats(features, 'rms', rms)
        del rms

        # ========================================================
        # HARMONIC / PERCUSSIVE SOURCE SEPARATION
        # ========================================================
        harmonic_stft, percussive_stft = librosa.decompose.hpss(stft)
        harmonic_stft_sq = harmonic_stft**2
        percussive_stft_sq = percussive_stft**2
        del harmonic_stft
        del percussive_stft
        harmonic_rms = np.sqrt(np.mean(harmonic_stft_sq, axis=0, keepdims=True))
        percussive_rms = np.sqrt(np.mean(percussive_stft_sq, axis=0, keepdims=True))
        _feature_stats(features, 'harmonic_rms', harmonic_rms)
        _feature_stats(features, 'percussive_rms', percussive_rms)
        del harmonic_rms
        del percussive_rms
        
        # Relative harmonic energy
        harmonic_energy = np.sum(harmonic_stft_sq, axis=0)
        percussive_energy = np.sum(percussive_stft_sq, axis=0)        
        total_energy = harmonic_energy + percussive_energy
        harmonic_ratio = (
            harmonic_energy / np.maximum(total_energy, np.finfo(np.float32).eps)
        )[np.newaxis, :]
        percussive_ratio = (
            percussive_energy / np.maximum(total_energy, np.finfo(np.float32).eps)
        )[np.newaxis, :]
        _feature_stats(features, 'harmonic_ratio', harmonic_ratio)
        _feature_stats(features, 'percussive_ratio', percussive_ratio)
        
        del harmonic_stft_sq
        del percussive_stft_sq
        del harmonic_energy
        del percussive_energy
        del total_energy
        del harmonic_ratio
        del percussive_ratio

        # ========================================================
        # SPECTRAL CENTROID
        # ========================================================
        spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
        _feature_stats(features, 'spectral_centroid', spectral_centroid)
        del spectral_centroid

        # ========================================================
        # SPECTRAL BANDWIDTH
        # ========================================================
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
        _feature_stats(features, 'spectral_bandwidth', spectral_bandwidth)
        del spectral_bandwidth

        # ========================================================
        # SPECTRAL FLUX
        # ========================================================
        spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0, keepdims=True))
        _feature_stats(features, 'spectral_flux', spectral_flux)
        del spectral_flux

        # ========================================================
        # SPECTRAL CONTRAST
        # ========================================================
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6)
        del stft  # no longer needed
        _feature_stats(features, 'spectral_contrast', spectral_contrast)
        del spectral_contrast

        # ========================================================
        # SPECTRAL FLATNESS
        # ========================================================
        spectral_flatness = librosa.feature.spectral_flatness(S=power_stft)
        _feature_stats(features, 'spectral_flatness', spectral_flatness)
        del spectral_flatness

        # ========================================================
        # SPECTRAL ROLLOFF
        # ========================================================
        spectral_rolloff_50 = librosa.feature.spectral_rolloff(S=power_stft, sr=sr, roll_percent=0.50)
        spectral_rolloff_85 = librosa.feature.spectral_rolloff(S=power_stft, sr=sr, roll_percent=0.85)
        spectral_rolloff_95 = librosa.feature.spectral_rolloff(S=power_stft, sr=sr, roll_percent=0.95)
        _feature_stats(features, 'spectral_rolloff_50', spectral_rolloff_50)
        _feature_stats(features, 'spectral_rolloff_85', spectral_rolloff_85)
        _feature_stats(features, 'spectral_rolloff_95', spectral_rolloff_95)
        del spectral_rolloff_50
        del spectral_rolloff_85
        del spectral_rolloff_95

        # ========================================================
        # MFCC + DELTAS
        # ========================================================
        # Reuses power_stft rather than recomputing an STFT.
        mel = librosa.feature.melspectrogram(S=power_stft, sr=sr, hop_length=hop_length)
        del power_stft  # no longer needed
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=20)
        _feature_stats(features, 'mfcc', mfcc)
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        _feature_stats(features, 'delta_mfcc', delta_mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        _feature_stats(features, 'delta2_mfcc', delta2_mfcc)
        del mfcc
        del delta_mfcc
        del delta2_mfcc

        # ========================================================
        # ONSET STRENGTH
        # ========================================================
        onset_strength = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
        _feature_stats(features, 'onset_strength', onset_strength)

        # ========================================================
        # TEMPOGRAM
        # ========================================================        
        tempogram = librosa.feature.tempogram(onset_envelope=onset_strength, sr=sr, hop_length=hop_length)
        tempogram_mean = np.mean(tempogram, axis=1)
        _feature_stats(features, 'tempogram_mean', tempogram_mean)
        del tempogram
        del tempogram_mean
        
        # ========================================================
        # BEAT TRACKING
        # ========================================================
        # Reuses the onset envelope computed above.
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_strength, sr=sr, hop_length=hop_length)
        del onset_strength  # no longer needed

        # Tempo
        tempo = float(np.asarray(tempo).ravel()[0])
        _feature_scalar(features, 'tempo', tempo)

        # Beat intervals
        if len(beats) >= 2:
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            beat_intervals = np.diff(beat_times)
            _feature_stats(features, 'beat_interval', beat_intervals)

        # Beat rate
        duration = len(x) / sr
        if duration > 0:
            beat_rate = len(beats) / duration
            _feature_scalar(features, 'beat_rate', beat_rate)

        del x  # no longer needed

    except Exception as exc:
        print(f"Failed to compute features for {tid}: {exc}")

    return features


def main() -> None:
    """Extract audio features for all tracks and save them to disk.

    Loads track metadata, optionally restricts it to a debug subset
    and/or a maximum number of tracks, then extracts features in
    parallel (grouped by track duration, to limit memory usage on long
    tracks), checkpointing progress to ``FEATURES_PATH`` along the way.
    Finally, reports and saves the list of tracks that failed
    extraction.
    """
    print(f"Audio files read from {AUDIO_DIR}.")
    print(f"Tracks loaded from {TRACKS_PATH}.")
    print(f"Features saved to {FEATURES_PATH}.")
    tracks = utils.load(TRACKS_PATH)
    if DEBUG_SUBSET:
        print(f"Selecting subset: {DEBUG_SUBSET}.")
        tracks = tracks[tracks['set', 'subset'] <= DEBUG_SUBSET]
    if DEBUG_LIMIT:
        print(f"Number of tracks to process is limited to {DEBUG_LIMIT}.")
        tracks = tracks[:DEBUG_LIMIT]

    features = pd.DataFrame(index=tracks.index, columns=columns(), dtype=np.float32)
    failed_tids = []

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    max_workers = int(1.5 * len(os.sched_getaffinity(0)))

    # Longest is ~11,000 seconds. Limit processes to avoid memory errors.
    table = ((5000, 1), (3000, 3), (2000, 5), (1000, 10), (0, max_workers))
    assert list(table) == sorted(table, reverse=True), "table must be sorted by descending duration"

    # Small improvement: safer and cleaner to use pool as context manager,
    # because it handles both pool.close() and pool.join() automatically.
    for duration, nb_workers in table:
        print('Working with {} processes.'.format(nb_workers))

        tids = tracks[tracks['track', 'duration'] >= duration].index
        tracks.drop(tids, axis=0, inplace=True)

        with multiprocessing.Pool(nb_workers) as pool:
            it = pool.imap_unordered(compute_features, tids)
            for i, row in enumerate(tqdm(it, total=len(tids))):
                features.loc[row.name] = row
                if i % 1000 == 0:
                    # This can be very costly since it rewrites the whole file
                    # again each time. If run time is too long, increase the
                    # checkpoint interval, or only write the new rows to the file.
                    save(features, 10)
                if row.isnull().all():
                    print(f"Failed to extract {row.name}.")
                    failed_tids.append(row.name)

    failed_tids = np.sort(failed_tids)
    np.save(FAILED_TIDS_PATH, failed_tids)
    if len(failed_tids) > 0:
        print(f"Data extraction failed for {len(failed_tids)} audios, corresponding to these audio IDs:\n{failed_tids}")
    save(features, 10)
    test(features, 10)


if __name__ == "__main__":
    main()
