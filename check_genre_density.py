import numpy as np
import utils
import os
from pathlib import Path


DEFAULT_DATA_DIR = os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/MusiClassi')
SUBSET_NAME = 'medium'
SUBSET_SELECTION = False

meta = Path(DEFAULT_DATA_DIR) / 'fma_metadata'
tracks = utils.load(meta / 'tracks.csv')

genre = tracks['track', 'genre_top']
labels = genre.astype(str)
artist = tracks['artist', 'id']

if SUBSET_SELECTION:
    subset = tracks['set', 'subset'] <= SUBSET_NAME
    tracks = tracks[subset]
    genre = genre[subset]
    artist = artist[subset]

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'
train_val = (tracks['set', 'split'] == 'training') | (tracks['set', 'split'] == 'validation')


if SUBSET_SELECTION:
    print(f"##### For subset == {SUBSET_NAME} #####")
else:
    print("##### For all tracks #####")

print(f"Number of tracks: {len(tracks)}")
print(f"Number of tracks with a top genre: {len(labels.dropna())}")

# for l in labels.unique():
#     print(f"Genre {l}:",
#           # f"{(genre[train]==l).sum()} over {genre[train].size} in train ---",
#           # f"{(genre[test]==l).sum()} over {genre[test].size} in test ---",
#           # f"ratio: {(genre[train]==l).sum() / genre[train].size} for train",
#           # f"and {(genre[test]==l).sum() / genre[test].size} for train"
#           f"\n    - {(genre[train]==l).sum() / genre[train].size * 100 :.2f}% in train",
#           f"\n    - {(genre[test]==l).sum() / genre[test].size * 100 :.2f}% in test",
#           f"\n    - {(genre[val]==l).sum() / genre[val].size * 100 :.2f}% in val",
#           # f"({(genre[train]==l).sum()} in {genre[train].size} for train",
#           # f"{(genre[test]==l).sum()} in {genre[test].size} for test)"
#           )

# for g in genre[subset].unique():
#     msk = (genre == g)
    # for a in artist[test & msk].unique():
    #     print(f"Genre: {g} ; Artist: {a} ; Number of tracks: {(artist[test & msk] == a).sum()}")

for l in sorted(labels.dropna().unique()):
    msk = (genre == l)
    print(f"Genre {l}:")
    print(f"  Total number of tracks: {(tracks['track', 'genre_top']==l).sum()}.")
    print(f"  Total number of tracks for train+val: {(tracks[train_val]['track', 'genre_top']==l).sum()}.")
    for s, name in zip([train, val, test], ['train', 'validation', 'test']):
        print(f"  - in {name}:")
        print(f"    -- Proportion: {(genre[s]==l).sum() / genre[s].size * 100 :.2f}%")
        print(f"    -- Number of artists: {(artist[s & msk]).unique().size}")
        print(f"    -- Number of tracks: {(tracks[s]['track', 'genre_top']==l).sum()}")

          # f"\n    - Proportion in val: {(genre[val]==l).sum() / genre[val].size * 100 :.2f}%",
          # f"\n    - Number of artists in val: {(artist[val & msk]).sum()}",
          # f"\n    - Number of tracks in val: {(tracks[val]['track', 'genre_top']==l).sum()}",          
          # f"\n    - Proportion in test: {(genre[test]==l).sum() / genre[test].size * 100 :.2f}%",
          # f"\n    - Number of artists in test: {(artist[test & msk]).sum()}",
          # f"\n    - Number of tracks in test: {(tracks[test]['track', 'genre_top']==l).sum()}",          
          # )

for s, name in zip([train, val, test], ['train', 'validation', 'test']):
    print("\n")
    for g in genre.unique():
        msk = (genre == g)
        n = 0
        for a in artist[s & msk].unique():
            if (artist[s & msk] == a).sum() > n:
                n = (artist[s & msk] == a).sum()
        print(f"[{name}] - Genre {g} dominated by artist {a} at {n*100/artist[s & msk].size :.2f}% in {name} set.")

