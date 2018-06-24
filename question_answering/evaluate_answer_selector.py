
import sys
import numpy as np
import pickle
import keras 

import sklearn
import sklearn.preprocessing
import sklearn.metrics 

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

import train_answer_selector 
from collections import defaultdict

import operator

    
import pdb
## 
THRESHOLD = 0.5 
##

def evaluate(model,  question_ids, candidate_ids,  test_labels, test_questions, test_answers) :
    print("Evaluating QA pair level scores")
    predicted_scores = model.predict( [test_questions, test_answers] )
    #it is predicted probabilities for both classes 0 and 1 . 
    #my interest is in only for class type 1, i.e if the pair is valid
    predicted_scores = predicted_scores[:, 1]
    ##

    ##  
    ## evaluate performance (prec, recall) for each question
    ## aggregate over dataset to get Mean. Avg. Precision, and Mean. Reciprocal. Rank
    print("Computing Performance Metrics:")
 

    true_y_per_question=defaultdict(list)
    pred_y_per_question=defaultdict(list)
    candidates_per_question = defaultdict(list)

    for idx, question_id in enumerate( question_ids ):
        predicted_score = predicted_scores[idx]
        true_label      = test_labels[idx][1] # test_labels is one-hot encoded with [0,1] or [1,0] to indicate valid or invalid which is 1,0
        candidate_id    = candidate_ids[idx]

        true_y_per_question[question_id].append(true_label)
        pred_y_per_question[question_id].append(predicted_score)
        candidates_per_question[question_id].append(candidate_id)
    ###
    mean_prec, mean_rr = 0.0, 0.0 #mean average precision and mean reciprocal rank
    valid_questions = 0.0 # a valid question is one with atleast one correct answer

    ##
    pdb.set_trace()
    for question_id in set( question_ids ):
        true_y           = true_y_per_question[question_id]
        candidate_ids    = candidates_per_question[question_id]
        predicted_scores = pred_y_per_question[question_id]
        predicted_labels = [ 1 if score > THRESHOLD else 0 for score in predicted_scores]
       
        ## find the correct candidates from true labels
        correct_candidates = [candidate_ids[idx] for idx,true_label in enumerate(true_y) if true_label ==1.0]  

        #skip all invalid questions for evaluation
        if len(correct_candidates) <= 0 : # if there is no correct answer :
            continue 
        else:
            valid_questions += 1.0
        ####
 
        ##
        average_precision = sklearn.metrics.average_precision_score (true_y, predicted_scores)
        mean_prec  += average_precision
        ##
        ## find the rank of correct candidates in the predicted scores
        # find candidate_ranking as per predicted scores  - order of scores in p corresponds to candidate id
        candidate_scores_sorted = np.argsort(predicted_scores)
        #map the candidates in ascending-order-positions to rank: position p -> rank (n-p) for p in (0,n-1)
        candidate_id_to_rank= { candidate_ids[idx] : len(candidate_scores_sorted) - candidate_scores_sorted[idx] 
                                for idx in range(len(candidate_scores_sorted)) } 

        #ranks of correctly retrieved answers
        ranks  = [ candidate_id_to_rank[candidate_id]  for candidate_id in correct_candidates]
        reciprocal_rank = 1.0 / min(ranks) # Reciprocal rank of the earliest and correct retrieval
        ##
        mean_rr +=  reciprocal_rank 
    ##
    no_of_questions = len( set(question_ids) )
    print("The Mean Average Precision is ",  mean_prec / valid_questions )
    print("The Mean Reciprocal Rank is "  ,  mean_rr / valid_questions )
 

############################################
if __name__ == "__main__" :
    ##  

    #tokenizer
    with open('data/tokenizer.pk', 'rb') as handle:
        tokenizer = pickle.load(handle)
    ##
    
    from keras.models import load_model
    model = load_model("data/answer_selector_model.h5")

    ## evaluate over test set
    question_ids, candidate_ids, test_labels, test_questions, test_answers   =  train_answer_selector.qa_pairs_to_xy( "data/test_pairs.txt" , tokenizer)    
    evaluate(model, question_ids, candidate_ids, test_labels, test_questions, test_answers)

     
    
    





    
    
        

        
   
