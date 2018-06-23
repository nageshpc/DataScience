
#A model M that assigns a score  in [0,1] for a Q-A pair , where Q is a question and A is a candidate answer.
# if M(Q-A) > \theta, A is considered a valid answer to Q , with \theta being a learnt threshold parameter.


import pandas as pd
import numpy as np
import pickle
import keras 
from keras.preprocessing.text import Tokenizer


#############################
def prepare_datasets( tsv_file ):
    ###split in 70:10:20 ratio for train:dev:test splits
    partitions = ["train", "val", "test"]
    try:
        partition_file = {  partition_name : open( "data/" + partition_name + "_pairs.txt", "w") for partition_name in partitions }
    except:
        print("check file open permissions", file=sys.stderr)
        sys.exit(-1)
    #################
    train_text = []
    tsv_data = pd.read_csv(tsv_file, sep="\t", encoding="ISO-8859-1")
    #partition 
    for question_id, candidate_set in tsv_data.groupby("QuestionID") :
        partition = np.random.choice( partitions, 1, replace=True, p=[0.7,0.1,0.2] )[0]
        ##
        if partition == "train" : 
            train_text.extend( candidate_set.Question.tolist() )
            train_text.extend( candidate_set.Sentence.tolist() )
        ##            
        for candidate in candidate_set.itertuples():
            print( question_id, candidate.SentenceID, candidate.Label
                    , candidate.Question.replace("\t", " ").replace("\n", " ")
                    , candidate.Sentence.replace("\t", " ").replace("\n", " ")
                    , sep="\t", file= partition_file[partition] )
        ##
    ## 
    for fh in partition_file.values():
        fh.close()   
    #################
    tokenizer= Tokenizer(num_words= 51000) # the actual vocabulary size is around 50,827
    tokenizer.fit_on_texts( train_text ) 

    # saving
    with open("data/tokenizer.pk", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##
########################################################
if __name__ == "__main__":
    np.random.seed(99) # Choosing a fixed seed for code reproducibility
    prepare_datasets("data/zendesk_challenge.tsv") 
