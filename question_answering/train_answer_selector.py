
import sys
import numpy as np
import pickle
import keras 

import sklearn
import sklearn.preprocessing

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten , Dropout, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Merge, LSTM, Dense , Masking
from keras.layers import TimeDistributed
from keras.preprocessing import sequence


###############
## A pair-classifier with 1D convolutions 

MAX_SEQ_LEN = 256  #  mean/median is around 22, but max is close to 247.
BATCH_SIZE = 32
EPOCHS  = 20
 

############################################################################################
def get_embedding_matrix(word_index):
    glove_file  = "./data/glove.6B.300d.txt"
    embedding_dimensionality = 300 

    # prepare embedding matrix
    num_words = max(word_index.values() ) +1   
    embedding_matrix = np.random.uniform(-0.25,0.25, (num_words, embedding_dimensionality) ).astype("float32")
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(glove_file) as f:  
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    ###
    print('Found %s word vectors. in the glove file' % len(embeddings_index))

    no_of_words_found_in_em_index= 0 
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None :   embeddings_index.get(word.lower()) # try lower case 
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            no_of_words_found_in_em_index +=1 
        else:
            print("did not find this word in the index ", word , file=sys.stderr)
    ##   
    print ("Embedding matrix ready. Its shape is", embedding_matrix.shape
            , " and the vocabulary size is ",  num_words 
            , " and the number of words with pre-trained embeddings is ", no_of_words_found_in_em_index
            )
    return embedding_matrix

#####################################################
def learn_answer_selector( word_index 
                            , train_labels, train_questions, train_answers
                            , val_labels, val_questions, val_answers) :

    num_words= max( word_index.values()) + 1
    num_classes = 2
    

    #get word embeddings
    word_embedding_matrix = get_embedding_matrix(word_index)
    embedding_dim = word_embedding_matrix.shape[1]
    
    #Keras Sequential model 
    word_embedding_layer= Embedding(num_words, embedding_dim, weights= [word_embedding_matrix] 
                                    , input_length = MAX_SEQ_LEN, trainable= False) 

    ## Layers of the NN 
    question_input  = Input( shape=(MAX_SEQ_LEN,), dtype='float32')     
    answer_input    = Input( shape=(MAX_SEQ_LEN,), dtype='float32')     
    question_embedding  = word_embedding_layer(question_input)
    answer_embedding    = word_embedding_layer(answer_input)

    ##conv 1d + max pooling (with  cell size 128, 3 or trigram window)
    ques_conv = Conv1D(128, 3, activation="relu")(question_embedding)
    ques_max = MaxPooling1D(5)(ques_conv)
    ques_vec = Flatten()(ques_max)

    ans_conv = Conv1D(128, 3, activation="relu")(answer_embedding)
    ans_max = MaxPooling1D(5)(ans_conv)
    ans_vec = Flatten()(ans_max)

    #merge the vector pairs + Dense layer + Logistic Regression as the final classifier
    merged_vector = keras.layers.concatenate( [ques_vec, ans_vec] ) 
    final_vector =  Dense(32, activation="relu")(merged_vector)

    ##
    predictions  =  Dense(num_classes, activation="softmax")(final_vector)
    model       =   Model([question_input, answer_input], predictions)
    ##
    print("Model summary ", model.summary() ) 
    print("Training the model ")
    model.compile( loss= "categorical_crossentropy", optimizer="adam", metrics=["acc"] )
    model.fit( x=[train_questions, train_answers] , y= train_labels 
                , validation_data= [ [val_questions, val_answers], val_labels] 
                , batch_size = BATCH_SIZE, epochs= EPOCHS)
    ##
    return model    

#####################################################
def qa_pairs_to_xy( data_file , tokenizer) :

    ####
    question_ids, candidate_ids, labels , questions, answers =[], [], [], [], []
    for qapair in open(data_file) :
        question_id, candidate_id, label , question, answer=    qapair.split("\t") 
        labels.append( int(label))
        questions.append(question)
        answers.append(answer)
        question_ids.append(question_id)
        candidate_ids.append(candidate_id)
    ####
    yvec = np.array(labels).reshape(-1,1)
    yvec = sklearn.preprocessing.OneHotEncoder(sparse=False).fit_transform(yvec) #make it [0,1] or [1,0]

    ##print('There were %s unique tokens.' % len(word_index), " but only top ", MAX_NB_WORDS , " will be used")
    que_sequence  = sequence.pad_sequences( tokenizer.texts_to_sequences(questions)  , maxlen= MAX_SEQ_LEN)
    ans_sequence  = sequence.pad_sequences( tokenizer.texts_to_sequences(answers)    , maxlen= MAX_SEQ_LEN)
    ##
    return question_ids, candidate_ids, yvec, que_sequence, ans_sequence

############################################
if __name__ == "__main__" :
    ######
    #tokenizer
    with open('data/tokenizer.pk', 'rb') as handle:
        tokenizer = pickle.load(handle)
    ##
    word_index = tokenizer.word_index
    

    
    ## learning  from training and validation sets
    _ , _ ,  train_labels, train_questions, train_answers   =  qa_pairs_to_xy( "data/train_pairs.txt" , tokenizer)    
    _ , _ ,  val_labels,   val_questions  , val_answers     =  qa_pairs_to_xy( "data/val_pairs.txt"   , tokenizer)        
    model = learn_answer_selector( word_index
                                   , train_labels, train_questions, train_answers
                                   , val_labels, val_questions, val_answers)
    ##
    model.save("data/answer_selector_model.h5")

     
    
    





    
    
        

        
   
