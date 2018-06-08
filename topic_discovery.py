

import pandas as pd
import spacy
import sys
import numpy as np
import matplotlib
import matplotlib.pylab as plt
### %matplotlib inline
from collections import defaultdict
import re
import sys
sys.path.append('/home/nagesh/anaconda/envs/dlenv/lib/python3.5/site-packages/')

import numpy as np
import nltk
import gensim

import operator
import string
import pdb


#nltk.download('stopwords')
#en_stop = set(nltk.corpus.stopwords.words('english'))
#############################################################
#def tokenize(text):
#    lda_tokens = []
#    from spacy.lang.en import English
#    parser = English()
#    tokens = parser(text )
#    import spacy 
#    spacy.load("en")
#    nlp.vocab   
#    for token in tokens:
#        if token.orth_.isspace():
#            continue
#        elif token.lower_ in 
#        elif token.like_url:
#            lda_tokens.append('URL')
#        #elif token.orth_.startswith('@'):
#        #    lda_tokens.append('SCREEN_NAME')
#        else:
#            lda_tokens.append(token.lower_)
#    #return lda_tokens
#    return [ get_lemma(tok) for tok in lda_tokens]
#
############################################################
def tokenize(text):
    from nltk import sent_tokenize
    from nltk import word_tokenize
    from nltk.corpus import stopwords

    #split on http 
    text = text.lower()
    text = text.replace("http://", " ")
    text  = "".join( [ ' ' if ch in string.punctuation else ch for ch in text] )


    tokens= word_tokenize(text )
    filtered_tokens= []
    stop_words = stopwords.words('english')
    for tok in tokens:
        if tok in ['https' , 'rt', 'co', 'thanks'] :
            continue
        elif tok in stop_words:
            continue 
        else:
            filtered_tokens.append( get_lemma(tok) )
    ##
    return filtered_tokens 
 
     

#############################################################
def get_lemma(word):
    #nltk.download('wordnet')
    from nltk.corpus import wordnet as wn

    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
#############################################################
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

#############################################################
#############################################################



##################
def discover_entities(data):
    nlp = spacy.load('en_core_web_sm')
 
    #doc = nlp(u'San Francisco considers banning sidewalk delivery robots')


    entities=    defaultdict(int)
    for ft in data.fulltext :
        doc = nlp( str(ft) )
        for ent in doc.ents :
            #print ( ent.text, ent.label_) 
            entities[ent.label_ + "-" + ent.text ] +=1
    ##
    #sort by frequency
    for ent,freq in sorted(entities.items() , key=operator.itemgetter(1) , reverse=True) :
        #print("the entities of type ", etype , " are  :" )
        #for ent in entities[etype] :
        print(freq, "\t",  ent )
    #####################


###################################################################
def get_topic_model(data):
    #topic discovery
    #import nltk
    #nltk.download("wordnet")

    documents =  [ str(ft) for ft in data.fulltext ] 
    corpus_raw_text = " ".join( documents ) 
    docs_as_tokens =   [  tokenize(doc) for doc in documents ]

    from gensim import corpora
    dictionary = corpora.Dictionary( docs_as_tokens)
    corpus = [dictionary.doc2bow(toklst) for toklst in docs_as_tokens]
    ##import pickle
    ##pickle.dump(corpus, open('corpus.pkl', 'wb'))
    ##dictionary.save('dictionary.gensim')


    import gensim
    NUM_TOPICS = 6
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=50)
    print("the topics are ")
    for topic in topics:
        print(topic) 
    ##
    #now classify each tweet into the topic it belongs and dump 
    new_columns= { top: [] for top in range(NUM_TOPICS) }
    for idx, doc in enumerate(corpus) :
        #find tweet -to- topic association-score
        topic_scores= { top: 0 for top in range(NUM_TOPICS) }
        for top,score in  ldamodel[doc] : #[(0, 0.11048241), (5, 0.49373671), (9, 0.36985034)]
            topic_scores[top] = score
        ##
        for top, score in topic_scores.items() :
            new_columns[top].append(score)
        ##
    ##
    for top in range( NUM_TOPICS ):
        data["topic_score_" + str(top) ] = new_columns[top]
    ##'
    data.to_csv( "./data/tweets_with_topics.csv" , sep="\t")


########################################
if __name__ == "__main__" :
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    data = pd.read_csv('./data/tweets.csv', sep="\t", parse_dates=['created_at'], date_parser=dateparse, dtype={'hashtags':str})
    print("Read the data. Its columns are:\n   " , re.sub("[ ]+",":", str(data.dtypes).replace("\n", ",\t") ) )

    # topic discovery 
    ##discover_entities(data)
    get_topic_model(data)  ## dumps the topic information into data/tweets_with_topics.csv


    ##now user to topic -plot the association.
    topic_data= pd.read_csv("./data/tweets_with_topics.csv", sep="\t")
    get_user_id=  { }
    for uname in topic_data.poster_name : 
        if uname not in get_user_id :
            get_user_id[uname] = len( get_user_id )
    ##
    user_topic_matrix = np.zeros( len(get_user_id) * 6 ).reshape(6, -1) 

    num_topics= 6 
    for idx, row in topic_data.iterrows() :
        uid = get_user_id[ row.poster_name ]
        for tnum in range(num_topics): 
            tname = "topic_score_" + str(tnum)
            tscore = row[tname] 
            user_topic_matrix[tnum, uid] += tscore
    ## 
    print ("now plot it") 
    ###
    fig, axes = plt.subplots(figsize=(11,3) )
    plt.imshow(user_topic_matrix, cmap='Purples', interpolation='nearest', aspect='auto')
    plt.colorbar()
    xticks= sorted(get_user_id.items() , key=operator.itemgetter(1) )  #name, uid - uid sorted
    plt.xticks( np.arange(len(xticks) ) ,  [n for (n,i) in xticks] , rotation='vertical' )   
    plt.yticks( np.arange(num_topics) ,  ["Videos/Graphics" , "Discussion", "ML Community News", "NLP", "Google-Related", "Academic"]  ) 
    plt.title("Visualizing User-to-Topic association")
    fig.tight_layout()
    plt.savefig("data/User_to_topic_association.jpg")
    #plt.show()
     
        

