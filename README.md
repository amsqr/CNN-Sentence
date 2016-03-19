# CNN-Sentence
Unsupervised learning of language understanding by sentence composition from words vector.

The purpose of this model is using CNN to do unsupervised language understanding. The sentence understanding is evaluated by the sentence topic classification problem.

The model has 3 training steps (we are only showing the code of step 2 and 3 here). 

1. Train word vector and phrase vector using skip gram method using large corpus. For the phases with high frenquences, it is connected by hyphen and through back into corpus (e.g big apple will become big_apple). The connected phrase will be treated as single world and will be predited using its context. This step will result vector for single words as well as phrases. e.g. vec('new_york'),vec('new'). The vectors are pretrained and are read using inputdata_process.py

2. Trainig of CNN network. The training is based on phrase and their tokenized words. E.g. for the training based on a_big_apple, the input will be [vec('a'),vec('big'),vec('apple')] and the output will be vec('a_big_apple'). The input is padded with zero to the designed CNN input dimension and then feed into the CNN network. Muti filter, max pooling and dropout are used to train the CNN. The cossentropy of output and vec('a_big_apple') are calculated as the loss. 
We hope the model can learn how to compose sentences by observing how to compose phrases, which can come from difference part of the sentence. Thus this CNN is supposed to get a vector representation with semantic meaning of sentences. 


3. Traaing of sentence classifier. We use the CNN trained above to get a vector representation of a sentence. Then we train a classifer on top of it to classify sentence topics. 
