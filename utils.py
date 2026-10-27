import os
import pandas as pd
import ast
import librosa
from tqdm import tqdm
from pathlib import Path


def load(filepath: str | os.PathLike) -> pd.DataFrame:
    """Load CSV file, base on filename.

    Args:
        filepath: the path to the CSV file to read.

    Returns:
        The DataFrame containing the CSV data.
    """
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

def get_audio_path(audio_dir: str | os.PathLike, track_id: int) -> str:
    """Return the path to the audio file given the directory
    where the audio is stored and the track ID.

    Args:
        audio_dir: The root directory of all audio files.
        track_id: The track ID of the file to retrieve.

    Returns:
        The path to the audio files.
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def get_audios(audio_dir: str | os.PathLike, track_ids: pd.Index) -> pd.DataFrame:
    """Load all the audio files corresponding to the given track IDs and
    store them in a DataFrame.

    Args:
        audio_dir: The root directory of all audio files.
        track_ids: The list of all track IDs to read.

    Returns:
        The DataFrame containing the data.
    Raises:
        FileNotFoundError: If an audio file for one of the track IDs
        cannot be found.
    """
    audios = []
    samplerates = []
    for i in tqdm(track_ids):
        filename = get_audio_path(audio_dir, i)
        x, sr = librosa.load(filename, sr=None, mono=True)
        audios.append(x)
        samplerates.append(sr)

    data = {'audio': audios,
            'sample rate': samplerates}
    audio_df = pd.DataFrame(data, index=track_ids)
    return audio_df

def df_to_latex(df: pd.DataFrame, ndigits: int = 3, caption: str = None, label: str = None) -> str:
    """Convert any DataFrame to LaTeX table source and print it.
 
    General-purpose wrapper around :meth:`pandas.DataFrame.to_latex`: not
    tied to any particular table, so it works equally well on a feature
    table, a timing summary, or anything else in a ``DataFrame``.
 
    Args:
        df: The table to convert.
        ndigits: Number of digits to keep after the decimal point for
            float columns.
        caption: Optional LaTeX table caption.
        label: Optional LaTeX ``\\label`` for cross-referencing.
 
    Returns:
        The LaTeX source of the table, as also printed to stdout.
    """
    latex = df.to_latex(float_format='%.{}f'.format(ndigits), caption=caption, label=label)
    print(latex)
    return latex


def txt_to_latex(path: Path, ndigits: int = 3, caption: str = None, label: str = None, **read_csv_kwargs) -> str:
    """Load a DataFrame fro m a whitespace-formatted .txt file and convert it to LaTeX.
 
    Reads a text file holding a table in the format produced by
    ``print(df)`` (columns separated by whitespace, row labels in the
    first column), such as ``TIMING_STATS_PATH``, then delegates to
    :func:`df_to_latex`.
 
    Args:
        path: Path to the .txt file to read.
        ndigits: Number of digits to keep after the decimal point for
            float columns.
        caption: Optional LaTeX table caption.
        label: Optional LaTeX ``\\label`` for cross-referencing.
        **read_csv_kwargs: Extra keyword arguments forwarded to
            :func:`pandas.read_csv` (e.g. ``sep`` or ``index_col``, if the
            file's layout differs from the default whitespace-separated,
            first-column-as-index format).
 
    Returns:
        The LaTeX source of the table, as also printed to stdout.
    """
    read_csv_kwargs.setdefault('sep', r'\s+')
    read_csv_kwargs.setdefault('index_col', 0)
    df = pd.read_csv(path, **read_csv_kwargs)
    return df_to_latex(df, ndigits=ndigits, caption=caption, label=label)
