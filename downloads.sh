# Download data
cd data
wget "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
tar xvzf rt-polaritydata.tar.gz

# Download word2vec embeddings
cd ../embeddings
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
