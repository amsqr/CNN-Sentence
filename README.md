# CNN-Sentence
Unsupervised learning of language understanding by sentence composition from words vector.
-Project to extract chemical synthesis graphs from academic papers, MIT 6.864 Fall 2015.
+Project to extract chemical synthesis graphs from academic papers, MIT
+6.864 Fall 2015.
+
+Running the Code
+================
+
+Running this code, in general, requires Theano, Keras, pymongo, h5py, and
+a number of other packages. I can only speak about my own code... - Quan
+
+First, you need to download and setup the chemnet database, which requires
+Mongo. After that's installed, run the mongo server daemon by doing:
+
+	$ mongod &
+
+Then, create the dataset by running:
+
+	$ ./create_datasets 1 100000
+
+This will create a file called `train_reactions_with_outputs_100000.json`,
+which contains 50,000 real reactions (with their outputs), and 50,000 fake
+reactions (with "null" as the output). (The code has been since modified
+to produce the output, users will need to convert the code so that it
+outputs true/false for reactions in order to perform the classification
+task. For your convenience, I have included )
+
+Assuming that the user has a file called `train_reactions_100000.json`,
+which contains [chemical A, chemical B, true/false] tuples, run the naive
+Bayes classifier to obtain performance:
+
+	$ ./naive.py train_reactions_100000.json
+
+To run the neural network model:
+
+	$ mkdir performance
+	$ ./neural.py train_reactions_100000.json
+
+This code creates a model that trains its own embeddings (and dumps them
+to a file called `embeddings_neural.p`, which is a dictionary that
+contains chemical name token to 100-dimensional embeddings). It also
+writes performance numbers to `performance/data_neural.tsv`.
+
+To run the neural network with fixed embeddings, first run the
+autoencoder:
+
+	$ ./autoencoder_3.py train_reactions_100000.json
+
+This creates a file called `embeddings_neural.p` as well, which is then
+fed to the neural network:
+
+	$ mkdir performance
+	$ ./neural_with_embeddings train_reactions_100000.json embeddings_neural.p
+
+It also places performance numbers in `performance/data.tsv`.
+
+Dropbox Code
+============
+
+Much of the group work was done on Dropbox. The MIT Dropbox link (only
+viewable by those at MIT) is:
+
+https://www.dropbox.com/sh/ls5d6jg0muuftzn/AACZlm8RIBlDIicvSQX7ybSka?dl=0
