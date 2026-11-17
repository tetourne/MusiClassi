import utils
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_DATA_DIR = os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/MusiClassi')
SUBSET_NAME = 'medium'
SUBSET_SELECTION = True
limit = 80  # number of genres to plot


def highlight_genres(ax, top_genres, color='red'):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if label.get_text() in top_genres:
            label.set_color(color)
            label.set_fontweight('bold')



meta = Path(DEFAULT_DATA_DIR) / 'fma_metadata'
tracks = utils.load(meta / 'tracks.csv')

raw_genres = pd.read_csv("/run/media/thomas/Data/Documents/MusiClassi/fma_metadata/raw_genres.csv")
genre_names = raw_genres.set_index('genre_id')['genre_handle']

genre_top = tracks['track', 'genre_top']
genres = tracks['track', 'genres']
artist = tracks['artist', 'id']

if SUBSET_SELECTION:
    subset = tracks['set', 'subset'] <= SUBSET_NAME
    tracks = tracks[subset]
    genre_top = genre_top[subset]
    genres = genres[subset]
    artist = artist[subset]

# train = tracks['set', 'split'] == 'training'
# val = tracks['set', 'split'] == 'validation'
# test = tracks['set', 'split'] == 'test'
# train_val = (tracks['set', 'split'] == 'training') | (tracks['set', 'split'] == 'validation')

top_genre_counts = genre_top.value_counts()
top_genres = set(top_genre_counts.index)

# Hist of number of genres per track
track_counts = [len(i) for i in genres]
plt.hist(track_counts, bins=np.max(track_counts)-np.min(track_counts)+1)
plt.title('number of genres per track')
plt.grid()
plt.tight_layout()

# Number of track per genres
genre_counts = genres.explode().value_counts()
genre_counts.index = genre_counts.index.map(genre_names)
genre_counts = genre_counts.head(limit)

# Horizontal plot
plt.figure(figsize=(16, 7))
genre_counts.sort_values(ascending=False).plot(kind='bar')

plt.xlabel('Genre')
plt.ylabel('Number of tracks')
plt.title(f'Top {limit} most populated genres')
plt.xticks(rotation=45, ha='right')
plt.grid()
highlight_genres(plt.gca(), top_genres)
plt.tight_layout()

# Vertical
plt.figure(figsize=(12, 8))
genre_counts.sort_values(ascending=True).plot(kind='barh')

plt.ylabel('Genre')
plt.xlabel('Number of tracks')
plt.title(f'Top {limit} most populated genres')
# plt.xticks(rotation=45, ha='right')
plt.grid()
highlight_genres(plt.gca(), top_genres)
plt.tight_layout()


# Count top-level genres
plt.figure(figsize=(12, 6))
top_genre_counts.plot(kind='bar')

plt.xlabel('Top-level genre')
plt.ylabel('Number of tracks')
plt.title('Distribution of top-level genres')
plt.xticks(rotation=45, ha='right')
plt.grid()
plt.tight_layout()


# Count genre pairs
pair_counts = Counter()
for genre in genres.dropna():
    # Remove duplicate genres within a track, if any
    genre = set(genre)

    # Convert IDs to names
    names = [genre_names.get(gid) for gid in genre]
    names = sorted(name for name in names if pd.notna(name))

    # Generate all unique pairs
    pair_counts.update(combinations(names, 2))

pair_df = pd.DataFrame(pair_counts.items(),columns=['Genre pair', 'Number of tracks'])
pair_df = pair_df.sort_values('Number of tracks', ascending=False).head(limit)

plt.figure(figsize=(12, 8))
plt.barh(
    pair_df['Genre pair'].apply(lambda x: ' + '.join(x))[::-1],
    pair_df['Number of tracks'][::-1]
)

ax = plt.gca()
for label in ax.get_yticklabels():
    pair = label.get_text().split(' + ')
    if any(genre in top_genres for genre in pair):
        label.set_color('red')
        label.set_fontweight('bold')

plt.xlabel('Number of tracks')
plt.ylabel('Genre pair')
plt.title(f'Top {limit} most frequent genre pairs')
plt.grid()
highlight_genres(plt.gca(), top_genres)
plt.tight_layout()


# Genres co-occurences
genres_per_track = genres.apply(
    lambda genres: [
        genre_names[gid]
        for gid in set(genres)
        if gid in genre_names.index
    ] if isinstance(genres, list) else []
)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(genres_per_track)

cooccurrence = pd.DataFrame(
    Y.T @ Y,
    index=mlb.classes_,
    columns=mlb.classes_
)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cooccurrence,
    cmap='Blues',
    annot=False
)

plt.title('Genre co-occurrence matrix')
plt.tight_layout()


plt.show()


