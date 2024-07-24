'''
This is the core of the project to "Discovering" articles' topics

use this to train an LDA model for the product

'''

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import multiprocessing
import pickle
import seaborn as sns

from matplotlib import pyplot as plt

class traininng_lda:
    def __init__(self,df_address,content_column_name,date_column_name,dictionary = None,use_multicore = True,passes = 20) -> None:
        self.df = self.loading_in_df(df_address)
        self.df = self.df[[content_column_name,date_column_name]]
        self.df[date_column_name] = pd.to_datetime(self.df[date_column_name]).dt.date
        self.multicore = use_multicore
        if dictionary:
            self.dictionary = dictionary
        else:
            self.dictionary,self.tokenized_contents = self.dictionize(contents=self.df[content_column_name])

        self.topicid_to_ids = None
        self.dictionary.save('dictionary.dict')

        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        self.lda_model = None
    
    def add_doc(self,doc):
        try:
            data = pd.read_csv(doc)
        except:
            data = doc
            pass

        data = self.preprocess(data)
        self.tokenized_contents.append(data)
        self.dictionary.add_documents(data)


    
    def loading_in_df(self,df_address):
        chunks = []
        for chunk in pd.read_csv(df_address,chunksize = 10000):
            chunks.append(chunk)
        loaded_df = pd.concat(chunks,ignore_index=True)
        return loaded_df
    
    def preprocess(self,text):
        text = text.lower()
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return words
    
    def dictionize(self,contents):
        contents = [self.preprocess(text) for text in contents]
        dictionary = Dictionary(contents)
        dictionary.filter_extremes(no_below =len(contents)/1000, no_above = 0.7)
        return dictionary, contents
    
    def Bow(self,dictionary,tokenized_contents):
        if len(tokenized_contents) > 1:

            bow = [dictionary.doc2bow(word) for word in tokenized_contents]
            return bow
        else:
            bow = dictionary.doc2bow(tokenized_contents)
            return bow
        
    def execute_training_lda(self,num_topics,passes = 20,num_workers = None,save_model = True,main_topics = True, evaluate = False):
        bow_corpus = self.Bow(self.dictionary,self.tokenized_contents)
        lda_model = LdaModel()
        if self.multicore:
            if num_workers == None:
                num_workers = multiprocessing.cpu_count() - 1  # Use all CPUs except one
            # sử dụng LDA multicore để chạy song song dữ liệu để chạy nhanh hơn
            lda_model = LdaMulticore(
                corpus=bow_corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=passes,
                workers=num_workers
            )
        else:
            lda_model = LdaModel(
                corpus=bow_corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                passes=self.passes,
            )
        self.lda_model = lda_model

        if save_model:
            lda_model.save(f'lda_model_{self.num_topics}.model')
        
        if main_topics:
            main_topic = self.bowList_to_mainTopicList(bow_corpus,lda_model,self.dictionary)
            self.df = pd.concat([self.df,main_topic],axis = 1)
            self.df.to_csv('topiced_sorted_df_all.csv', index= False)

            self.topicid_to_ids = self.topic_table(bow_corpus,lda_model)
    
    def topic_table(self,bows,lda_model,no_lower_than =0.2, num = 3,save = True):
        num_topics = lda_model.num_topics
        values = [[] for _ in range(num_topics)]

        for i in range(len(bows)):
            p_topics = lda_model.get_document_topics(bows[i], minimum_probability = no_lower_than)
            top = sorted(p_topics, key=lambda x: x[1], reverse=True)[:num]
            
            for topic_id, v in top:
                values[topic_id].append([i,v])

        topicid_to_ids = {}

        for i in range(len(values)):
            topicid_to_ids[i] = values[i]

        if save:
            with open(f'topicid_to_ids_{num_topics}.pkl', 'wb') as f:
                pickle.dump(topicid_to_ids, f)

        return topicid_to_ids
    


    def bowList_to_mainTopicList(self,listBow,ldamodel,dictionary):
        main_topic = []
        for doc in listBow:
            p_topics = ldamodel.get_document_topics(doc)
            topic = sorted(p_topics, key=lambda x: x[1], reverse=True)[:1]
            main_topic.append(topic[0][0])
        soTopic = ldamodel.num_topics
        main_topic = pd.DataFrame(main_topic, columns= [f'main_topic_{soTopic}'])
        return main_topic

    def self_similarity_score(self,lda_model):
        topics = lda_model.get_topics()
        n_topics = lda_model.num_topics
        similarity_matrix = cosine_similarity(topics,topics)
        
        similarity_matrix = np.array(similarity_matrix)
        np.fill_diagonal(similarity_matrix,-1)
        score = 999 ** similarity_matrix

        score = np.sum(score) /(n_topics**2)
        
        return score
    
    def plot_heatmap(lda_model, figsize=(100, 80)):
        annot=True
        fmt=".2f"
        cmap='coolwarm'
        title='Topic Similarity Matrix'
        xlabel='Model 2 Topics'
        ylabel='Model 1 Topics'

        topics = lda_model.get_topics()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cosine_similarity(topics,topics), annot=annot, fmt=fmt, cmap=cmap, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        return ax


