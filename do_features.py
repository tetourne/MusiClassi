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
import shutil
import subprocess
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
# Tracks with at least one NaN feature (superset of FAILED_TIDS_PATH).
NAN_TIDS_PATH = DATA_DIR / 'fma_metadata' / 'nan_tids_medium.npy'
# Fast binary checkpoint, written during extraction instead of the big CSV.
CHECKPOINT_PATH = DATA_DIR / 'fma_metadata' / 'myfeatures_checkpoint_medium.pkl'

# Float32 needs 9 significant digits to round-trip exactly; '%.8e' gives 9.
SAVE_DIGITS = 8
# Write a checkpoint every N tracks (binary, so cheap; the CSV is only
# written once, at the end).
CHECKPOINT_EVERY = 5000
# Sample rate used by the ffmpeg fallback decoder (see _load_audio).
FALLBACK_SR = 44100
# Minimum amount of audio the fallback must recover to be trusted.
MIN_FALLBACK_SECONDS = 5

# Benign: librosa falls back to 0 tuning when a (near-)silent track has no
# pitched peaks. Filtered at module level so worker processes inherit it.
warnings.filterwarnings(
    'ignore', message='Trying to estimate tuning from empty frequency set'
)

# Used for test and debug
DEBUG_LIMIT = None  # max number of files to load. Set to None if no use
DEBUG_SUBSET = 'small'


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


def save_checkpoint(features: pd.DataFrame) -> None:
    """Write a fast binary snapshot of the (unsorted) feature table.

    Much cheaper than :func:`save`, which formats millions of floats as text
    and rewrites the whole CSV. Use ``pd.read_pickle(CHECKPOINT_PATH)`` to
    recover partial results if a run is interrupted.

    Args:
        features: Table of extracted features so far.
    """
    features.to_pickle(CHECKPOINT_PATH)


def test(features: pd.DataFrame, ndigits: int) -> None:
    """Sanity-check the saved features against the in-memory table.

    Reports (and saves to ``NAN_TIDS_PATH``) any track with missing (NaN)
    values, then reloads the CSV written by :func:`save` and checks it
    matches ``features`` within tolerance.

    The comparison is done in float64 with a tiny absolute tolerance. In
    float32, ``rtol * |value|`` underflows to 0 for denormal values (around
    1e-45, typical of silent tracks), so *any* rounding difference introduced
    by the CSV text format would be reported as a mismatch.

    Args:
        features: Table of extracted features to compare against the
            saved CSV.
        ndigits: Number of significant digits used when the features were
            saved; used to derive the comparison tolerance.
    """
    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        np.save(NAN_TIDS_PATH, np.sort(indices.to_numpy()))
        print(f'{len(indices)} tracks have at least one NaN feature '
              f'(ids saved to {NAN_TIDS_PATH}).')

    tmp = utils.load(FEATURES_PATH)
    np.testing.assert_allclose(
        tmp.values.astype(np.float64),
        features.values.astype(np.float64),
        rtol=10**-ndigits,
        atol=1e-30,
    )
    print('Saved CSV matches the in-memory features.')


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


def _skew_kurtosis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Skewness and (Fisher) kurtosis along axis 1, robust to flat features.

    The moments are computed in float64. In float32 (librosa's dtype), SciPy
    reports "catastrophic cancellation" for features that are merely
    *almost* constant (quiet or silent passages, band-limited audio).
    Features that are constant up to float32 noise have no meaningful
    skewness or kurtosis, so they are set to 0 instead of NaN or garbage.

    Args:
        values: Array of shape ``(n_features, n_frames)``.

    Returns:
        Tuple ``(skew, kurtosis)``, each of shape ``(n_features,)``.
    """
    v = values.astype(np.float64)
    mean = v.mean(axis=1)
    std = v.std(axis=1)
    flat = std <= 1e-6 * np.abs(mean) + np.finfo(np.float32).tiny

    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        skew = np.asarray(stats.skew(v, axis=1), dtype=np.float64)
        kurt = np.asarray(stats.kurtosis(v, axis=1), dtype=np.float64)

    skew[flat] = 0.0
    kurt[flat] = 0.0
    return skew, kurt


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
    # float64 so that all statistics (not just skew/kurtosis) are accurate.
    values = np.atleast_2d(values).astype(np.float64)  # (n_features, n_frames)
    skew, kurt = _skew_kurtosis(values)

    stats_map = {
        'mean': np.mean(values, axis=1),
        'std': np.std(values, axis=1),
        'skew': skew,
        'kurtosis': kurt,
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


def _load_audio(filepath: Path, tid: int) -> tuple[np.ndarray, float]:
    """Load a track as mono float32, falling back to ffmpeg if needed.

    First tries ``librosa.load`` at the native sample rate. Some FMA mp3s
    are truncated or contain zero-filled headers, which libmpg123 (used by
    libsndfile) gives up on. ffmpeg is more tolerant of broken streams, so
    if it is installed it is used as a fallback. Fallback audio is resampled
    to ``FALLBACK_SR`` (the sample rate of the vast majority of FMA clips)
    because the native rate cannot be read from a broken file reliably.

    Args:
        filepath: Path to the audio file.
        tid: Track id, only used for the diagnostic message.

    Returns:
        Tuple ``(samples, sample_rate)``.

    Raises:
        Exception: The original ``librosa`` error if the fallback is
            unavailable or cannot decode enough audio either.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', module='librosa')
            return librosa.load(filepath, sr=None, mono=True)
    except Exception as exc:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise
        try:
            proc = subprocess.run(
                [ffmpeg, '-v', 'error', '-i', str(filepath),
                 '-f', 'f32le', '-acodec', 'pcm_f32le',
                 '-ac', '1', '-ar', str(FALLBACK_SR), '-'],
                capture_output=True, timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            raise exc
        x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if x.size < MIN_FALLBACK_SECONDS * FALLBACK_SR:
            raise exc
        print(f"Note: {tid} decoded with the ffmpeg fallback "
              f"({x.size / FALLBACK_SR:.1f} s of audio).")
        return x, FALLBACK_SR


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
        x, sr = _load_audio(filepath, tid)

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
    tracks), writing a binary checkpoint to ``CHECKPOINT_PATH`` every
    ``CHECKPOINT_EVERY`` tracks. Finally, reports and saves the list of
    tracks that failed extraction, and writes the CSV.
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
                if i > 0 and i % CHECKPOINT_EVERY == 0:
                    # Binary snapshot: fast. The (slow) CSV is written once,
                    # at the end.
                    save_checkpoint(features)
                if row.isnull().all():
                    print(f"Failed to extract {row.name}.")
                    failed_tids.append(row.name)

    failed_tids = np.sort(failed_tids)
    np.save(FAILED_TIDS_PATH, failed_tids)
    if len(failed_tids) > 0:
        print(f"Data extraction failed for {len(failed_tids)} audios, corresponding to these audio IDs:\n{failed_tids}")
    save_checkpoint(features)  # safety net in case the CSV write fails
    save(features, SAVE_DIGITS)
    test(features, SAVE_DIGITS)


if __name__ == "__main__":
    main()
