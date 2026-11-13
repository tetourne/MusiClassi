# MusiClassi
A tool to classify songs based on musical genres and generate playlists according to genres
This project is lead by Thomas Etourneau (thomas.etourneau@tutanota.com).
This is an ongoing project.

# State of the project
The FMA small dataset has been analysed. 
I performed my own feature extraction.
I lead several analysis, which are presented in different documents (in the analysis directory):
- a computational time analysis for the feature extraction
- a model comparison
- a feature correlation analysis
- an ablation analysis with the best models obtained.

I then moved to the FMA medium dataset.
The performances of the models dropped. Because some genres (from 8 to 16 genres in FMA medium) are very poorly represented.
I need to find a way to increase the performances on these genres. Probably by adding more data to the dataset, 


# Thanks
I would like to thanks the team behind the FMA project (https://github.com/mdeff/fma) which allowed me to perform this analysis and build my models and project with this amazing dataset.
