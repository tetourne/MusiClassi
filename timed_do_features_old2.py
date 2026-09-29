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
import time


# Paths
DATA_DIR = Path(os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/Database'))
AUDIO_DIR = DATA_DIR / "fma_small"
TRACKS_PATH = DATA_DIR / 'fma_metadata' / 'tracks.csv'
FEATURES_PATH = DATA_DIR / 'fma_metadata' / 'myfeatures.csv'
FAILED_TIDS_PATH = DATA_DIR / 'fma_metadata' / 'failed_tids.npy'
TIMINGS_PATH = DATA_DIR / 'fma_metadata' / 'timings.df'
TIMING_STATS_PATH = DATA_DIR / 'fma_metadata' / 'timing_stats.txt'

# Used for test and debug
DEBUG_LIMIT = 100 # max number of files to load. Set to None if no use
DEBUG_SUBSET = 'small'


def save(features, ndigits):

    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv(FEATURES_PATH, float_format='%.{}e'.format(ndigits))


def test(features, ndigits):

    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print('Failed tracks: {}'.format(', '.join(str(i) for i in indices)))

    tmp = utils.load(FEATURES_PATH)
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


def columns():
    feature_sizes = dict(
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        tonnetz_cqt=6,
        tonnetz_cens=6,
        mfcc=20,
        delta_mfcc=20,
        rms=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_flatness=1,
        spectral_rolloff_50=1,
        spectral_rolloff_85=1,
        spectral_rolloff_95=1,
        onset_strength=1,
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

    return columns.sort_values()

def compute_features(tid):
    """
    Compute audio features for one FMA track.

    Returns
    -------
    features : pandas.Series
        Extracted feature vector.

    timings : dict
        Time spent in the different computation stages.
    """

    filepath = utils.get_audio_path(AUDIO_DIR, tid)

    # ------------------------------------------------------------
    # Prepare output
    # ------------------------------------------------------------

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    timings = {}

    total_start = time.perf_counter()

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------

    def feature_stats(name, values):
        """
        Compute summary statistics over time.

        Expected shape:
            (n_features, n_frames)
        """

        stats_start = time.perf_counter()

        # Make sure all data are 2D (shaped as (n_features, n_frames))
        values = np.atleast_2d(values)

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

        # for moment, arr in stats_map.items():
        #     arr = np.asarray(arr, dtype=np.float32)
        #     for i, value in enumerate(arr):
        #         col = (name, moment,'{:02d}'.format(i + 1))
        #         if col in features.index:
        #             features.loc[col] = value

        return time.perf_counter() - stats_start

    def feature_scalar(name, value):
        """
        Store a scalar feature (single value, no distribution stats).
        """
        col = (name, 'value', '01')

        if col in features.index:
            features.loc[col] = np.float32(value)


    # ------------------------------------------------------------
    # Load audio
    # ------------------------------------------------------------

    start = time.perf_counter()

    try:

        with warnings.catch_warnings():

            warnings.filterwarnings('error', module='librosa')

            x, sr = librosa.load(filepath, sr=None, mono=True)

        timings['load_audio'] = (time.perf_counter() - start)

        # --------------------------------------------------------
        # Common parameters
        # --------------------------------------------------------

        hop_length = 512
        n_fft = 2048

        # ========================================================
        # ZERO CROSSING RATE
        # ========================================================

        start = time.perf_counter()

        zcr = librosa.feature.zero_crossing_rate(
            x,
            frame_length=n_fft,
            hop_length=hop_length
        )

        timings['zcr_compute'] = (
            time.perf_counter() - start
        )

        timings['zcr_stats'] = feature_stats(
            'zcr',
            zcr
        )

        # ========================================================
        # CQT
        # ========================================================

        # CQT is one of the potentially expensive operations.
        #
        # We compute it ONCE and reuse it for:
        #
        #   - chroma_cqt
        #   - chroma_cens
        #   - tonnetz_cqt
        #   - tonnetz_cens
        #

        start = time.perf_counter()

        cqt = np.abs(
            librosa.cqt(
                x,
                sr=sr,
                hop_length=hop_length,
                bins_per_octave=12,
                n_bins=7 * 12,
                tuning=None
            )
        )

        timings['cqt_compute'] = (
            time.perf_counter() - start
        )

        # ========================================================
        # CHROMA CQT
        # ========================================================

        start = time.perf_counter()

        chroma_cqt = librosa.feature.chroma_cqt(
            C=cqt,
            sr=sr,
            n_chroma=12,
            n_octaves=7
        )

        timings['chroma_cqt_compute'] = (
            time.perf_counter() - start
        )

        timings['chroma_cqt_stats'] = feature_stats(
            'chroma_cqt',
            chroma_cqt
        )

        # ========================================================
        # CHROMA CENS
        # ========================================================

        start = time.perf_counter()

        chroma_cens = librosa.feature.chroma_cens(
            C=cqt,
            sr=sr,
            n_chroma=12,
            n_octaves=7
        )

        timings['chroma_cens_compute'] = (
            time.perf_counter() - start
        )

        timings['chroma_cens_stats'] = feature_stats(
            'chroma_cens',
            chroma_cens
        )

        # ========================================================
        # TONNETZ FROM CQT CHROMA
        # ========================================================

        start = time.perf_counter()

        tonnetz_cqt = librosa.feature.tonnetz(
            chroma=chroma_cqt,
            sr=sr
        )

        timings['tonnetz_cqt_compute'] = (
            time.perf_counter() - start
        )

        timings['tonnetz_cqt_stats'] = feature_stats(
            'tonnetz_cqt',
            tonnetz_cqt
        )

        # ========================================================
        # TONNETZ FROM CENS CHROMA
        # ========================================================

        start = time.perf_counter()

        tonnetz_cens = librosa.feature.tonnetz(
            chroma=chroma_cens,
            sr=sr
        )

        timings['tonnetz_cens_compute'] = (
            time.perf_counter() - start
        )

        timings['tonnetz_cens_stats'] = feature_stats(
            'tonnetz_cens',
            tonnetz_cens
        )

        # CQT no longer needed
        del cqt

        # ========================================================
        # STFT
        # ========================================================

        # Compute magnitude STFT ONCE.
        #
        # It is reused for:
        #
        #   - chroma_stft
        #   - RMS
        #   - spectral centroid
        #   - spectral bandwidth
        #   - spectral contrast
        #   - spectral flatness
        #   - spectral rolloff
        #   - mel spectrogram
        #

        start = time.perf_counter()

        stft = np.abs(
            librosa.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length
            )
        )

        timings['stft_compute'] = (
            time.perf_counter() - start
        )

        # Compute power only once.
        start = time.perf_counter()

        power_stft = stft ** 2

        timings['stft_power'] = (
            time.perf_counter() - start
        )

        # ========================================================
        # CHROMA STFT
        # ========================================================

        start = time.perf_counter()

        chroma_stft = librosa.feature.chroma_stft(
            S=power_stft,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        timings['chroma_stft_compute'] = (
            time.perf_counter() - start
        )

        timings['chroma_stft_stats'] = feature_stats(
            'chroma_stft',
            chroma_stft
        )

        # ========================================================
        # RMS
        # ========================================================

        start = time.perf_counter()

        rms = librosa.feature.rms(
            S=stft,
            frame_length=n_fft,
            hop_length=hop_length
        )

        timings['rms_compute'] = (
            time.perf_counter() - start
        )

        timings['rms_stats'] = feature_stats('rms', rms)

        # ========================================================
        # SPECTRAL CENTROID
        # ========================================================

        start = time.perf_counter()

        spectral_centroid = (
            librosa.feature.spectral_centroid(
                S=stft,
                sr=sr
            )
        )

        timings['spectral_centroid_compute'] = (
            time.perf_counter() - start
        )

        timings['spectral_centroid_stats'] = feature_stats(
            'spectral_centroid',
            spectral_centroid
        )

        # ========================================================
        # SPECTRAL BANDWIDTH
        # ========================================================

        start = time.perf_counter()

        spectral_bandwidth = (
            librosa.feature.spectral_bandwidth(
                S=stft,
                sr=sr
            )
        )

        timings['spectral_bandwidth_compute'] = (
            time.perf_counter() - start
        )

        timings['spectral_bandwidth_stats'] = feature_stats(
            'spectral_bandwidth',
            spectral_bandwidth
        )

        # ========================================================
        # SPECTRAL CONTRAST
        # ========================================================

        start = time.perf_counter()

        spectral_contrast = (
            librosa.feature.spectral_contrast(
                S=stft,
                sr=sr,
                n_bands=6
            )
        )

        del stft  # not used anymore

        timings['spectral_contrast_compute'] = (
            time.perf_counter() - start
        )

        timings['spectral_contrast_stats'] = feature_stats(
            'spectral_contrast',
            spectral_contrast
        )

        # ========================================================
        # SPECTRAL FLATNESS
        # ========================================================

        start = time.perf_counter()

        spectral_flatness = (
            librosa.feature.spectral_flatness(
                S=power_stft
            )
        )

        timings['spectral_flatness_compute'] = (
            time.perf_counter() - start
        )

        timings['spectral_flatness_stats'] = feature_stats(
            'spectral_flatness',
            spectral_flatness
        )

        # ========================================================
        # SPECTRAL ROLLOFF
        # ========================================================

        start = time.perf_counter()

        spectral_rolloff_50 = (
            librosa.feature.spectral_rolloff(
                S=power_stft,
                sr=sr,
                roll_percent=0.50
            )
        )

        spectral_rolloff_85 = (
            librosa.feature.spectral_rolloff(
                S=power_stft,
                sr=sr,
                roll_percent=0.85
            )
        )

        spectral_rolloff_95 = (
            librosa.feature.spectral_rolloff(
                S=power_stft,
                sr=sr,
                roll_percent=0.95
            )
        )

        timings['spectral_rolloff_compute'] = (
            time.perf_counter() - start
        )

        stats_start = time.perf_counter()

        feature_stats('spectral_rolloff_50', spectral_rolloff_50)
        feature_stats('spectral_rolloff_85', spectral_rolloff_85)
        feature_stats('spectral_rolloff_95', spectral_rolloff_95)

        timings['spectral_rolloff_stats'] = (
            time.perf_counter() - stats_start
        )

        # ========================================================
        # MEL SPECTROGRAM
        # ========================================================

        # Reuse power_stft rather than recomputing an STFT.

        start = time.perf_counter()

        mel = librosa.feature.melspectrogram(S=power_stft, sr=sr, hop_length=hop_length)

        del power_stft  # Not used anymore

        timings['mel_compute'] = (
            time.perf_counter() - start
        )

        # ========================================================
        # MFCC
        # ========================================================

        start = time.perf_counter()

        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=20)

        del mel  # Not used anymore

        timings['mfcc_compute'] = (
            time.perf_counter() - start
        )

        timings['mfcc_stats'] = feature_stats(
            'mfcc',
            mfcc
        )

        # ========================================================
        # DELTA MFCC
        # ========================================================

        start = time.perf_counter()

        delta_mfcc = librosa.feature.delta(mfcc,order=1)

        del mfcc  # Not used anymore
        
        timings['delta_mfcc_compute'] = (
            time.perf_counter() - start
        )

        timings['delta_mfcc_stats'] = feature_stats(
            'delta_mfcc',
            delta_mfcc
        )

        del delta_mfcc  # Not used anymore

        # ========================================================
        # ONSET STRENGTH
        # ========================================================

        start = time.perf_counter()

        onset_strength = librosa.onset.onset_strength(
            y=x,
            sr=sr,
            hop_length=hop_length
        )

        timings['onset_strength_compute'] = (
            time.perf_counter() - start
        )

        timings['onset_strength_stats'] = feature_stats(
            'onset_strength',
            onset_strength
        )

        # ========================================================
        # BEAT TRACKING
        # ========================================================

        # Reuse the onset envelope computed above.

        start = time.perf_counter()

        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_strength,
            sr=sr,
            hop_length=hop_length
        )

        del onset_strength  # Not used anymore

        timings['beat_tracking_compute'] = (
            time.perf_counter() - start
        )

        # --------------------------------------------------------
        # Tempo
        # --------------------------------------------------------

        tempo = float(np.asarray(tempo).ravel()[0])
        feature_scalar('tempo', tempo)

        # --------------------------------------------------------
        # Beat intervals
        # --------------------------------------------------------

        stats_start = time.perf_counter()

        if len(beats) >= 2:
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            beat_intervals = np.diff(beat_times)
            feature_stats(
                'beat_interval',
                beat_intervals
            )

        timings['beat_interval_stats'] = (
            time.perf_counter() - stats_start
        )

        # --------------------------------------------------------
        # Beat rate
        # --------------------------------------------------------

        duration = len(x) / sr

        if duration > 0:
            beat_rate = len(beats) / duration
            feature_scalar('beat_rate', beat_rate)

        del x  # Not used anymore

    except Exception as exc:

        print(f"Failed to compute features for {tid}: {exc}")
        return features, timings

    # ============================================================
    # TOTAL TIME
    # ============================================================

    timings['total'] = (
        time.perf_counter() - total_start
    )

    return features, timings


def main():
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

    features = pd.DataFrame(index=tracks.index,
                            columns=columns(), dtype=np.float32)
    failed_tids = []
    timings = []

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

        for i, tid in enumerate(tqdm(tids)):
            row, timing = compute_features(tid)
            timings.append(timing)
            features.loc[row.name] = row
            if i % 1000 == 0:
                # this can be very cosly since it's rewriting the whole file
                # again each time. If run time is too long, increase the
                # checkpoint interval, or only write the new rows to the file.
                save(features, 10)
            if row.isnull().all():
                print(f"Failed to extract {row.name}.")
                failed_tids.append(row.name)

        ## Don't use multiprocessing for compute time
        # with multiprocessing.Pool(nb_workers) as pool:
        #     it = pool.imap_unordered(compute_features, tids)
        #     for i, tup in enumerate(tqdm(it, total=len(tids))):
        #         row, timing = tup
        #         timings.append(timing)                    
        #         features.loc[row.name] = row
        #         if i % 1000 == 0:
        #             # this can be very cosly since it's rewriting the whole file
        #             # again each time. If run time is too long, increase the
        #             # checkpoint interval, or only write the new rows to the file.
        #             save(features, 10)
        #         if row.isnull().all():
        #             print(f"Failed to extract {row.name}.")
        #             failed_tids.append(row.name)

    failed_tids = np.sort(failed_tids)
    np.save(FAILED_TIDS_PATH, failed_tids)
    timings_df = pd.DataFrame(timings)
    summary = timings_df.agg(['mean', 'std', 'min', 'max']).T.sort_values('mean', ascending=False)
    print(summary)
    with open(TIMING_STATS_PATH, 'w') as file:
        print(summary, file=file)
    timings_df.to_pickle(TIMINGS_PATH)
    if len(failed_tids)>0:
        print(f"Data extraction failed for {len(failed_tids)} audios, corresponding to these audio IDs:\n{failed_tids}")
    save(features, 10)
    test(features, 10)


if __name__ == "__main__":
    main()
