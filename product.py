'''
this is the final product of the project, with the core is the results taken after the training process with the datasets taken from AXVIR
currently the model using for the final product is an LDA model with the number of topic is 80

the Lda Model used to get the information can be change but one will need to single-handedly train and create lookup tables (topicid_to_ids : for Article Search, topic_trend_day : for Trend Analysis).


the product has 3 function:

Showing trends within a given timeframe


'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pathlib

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
from gensim.models import LdaModel
import pickle
from wordcloud import WordCloud



class blind_search_engine:

    def __init__(self):

        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.lda_model, self.df, self.dictionary, self.topicid_to_ids = self.load()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        

        
    




    def preprocess(self, text):
        text = text.lower()
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return words

    def load(self):
        chunks = []

        df_address = str(pathlib.Path(__file__).parent.resolve()) + '/saved_data/sorted_df.csv'


        for chunk in pd.read_csv(df_address, chunksize=10000):
            chunks.append(chunk)

        loaded_df = pd.concat(chunks, ignore_index=True)


        dict_address = str(pathlib.Path(__file__).parent.resolve()) + '/saved_data/dictionary.dict'
        loaded_dictionary = Dictionary.load(dict_address)


        model_address = str(pathlib.Path(__file__).parent.resolve()) + '/saved_data/lda_model_80.model'
        loaded_lda_model = LdaModel.load(model_address)

         #topicid_toids_address = str(pathlib.Path(__file__).parent.resolve()) + '/saved_data/topicid_to_ids.pkl'
        topicid_toids_address = str(pathlib.Path(__file__).parent.resolve()) + '/saved_data/new_topicid_to_ids_80.pkl'
        with open(topicid_toids_address, 'rb') as f:
            loaded_topicid_to_ids = pickle.load(f)

        return loaded_lda_model, loaded_df, loaded_dictionary, loaded_topicid_to_ids

    def bow_to_topicid(self, bow, no_lower_than=0.2, num=3):
        p_topics = self.lda_model.get_document_topics(bow, minimum_probability=no_lower_than)
        top = sorted(p_topics, key=lambda x: x[1], reverse=True)[:num]
        return top

    def search_to_topicid(self, search, no_lower_than=0.2, num=3):
        preprocess_search = self.preprocess(search)
        bow_search = self.dictionary.doc2bow(preprocess_search)
        top = self.bow_to_topicid(bow_search, no_lower_than, num)
        return top

    def id_to_contents(self, id):
        result = self.df.loc[self.df['id'] == id]
        return result

    def search_results(self, top, num_result=10, topics=3):

        results = pd.DataFrame()


        for topicid, prob_ in top[:topics]:
            ids = self.topicid_to_ids[topicid]
            probs = []
            for id,prob in ids:
                next = self.id_to_contents(id)
                probs.append(prob)
                if num_result == 0:
                    break
                elif next is None:
                    break
                else:
                    num_result = num_result - 1
                
                results = pd.concat([results, next], axis=0)

        return results,probs



    def search(self, text: str, no_lower_than=0.2, num=3, num_result=10, topics=3):
        top = self.search_to_topicid(text, no_lower_than, num)
        results,probs = self.search_results(top, num_result, topics)
        return results,probs
    
    def turnOn(self,no_lower_than=0.2,num_result=10,topics=3):
        x = input('Search >>: ')
        while x != 'off' and x != 'end' and x!= 'close':
            
            results = self.search(x,no_lower_than,num=3,num_result=10,topics=3)
            print(results)
            x = input('Search >>: ')
        print('Search Ended')
    

def create_topic_trend_table(df,n_topic,save = False):
    topic_trend = 0
    #cái try này ko quan trọng lắm, vì t để máy của bọn m và máy t khác nhau nên để thế thôi
    try:
        topic_trend = df.groupby(['main_topic', 'int_dates']).size().to_frame().unstack()
    except:
        topic_trend = df.groupby(['main_topic_80', 'int_dates']).size().to_frame().unstack()
    topic_trend = topic_trend.fillna(0)
    #soTopic = len(df['main_topic'].unique())
    tempt = topic_trend.loc[:,0]
    if save:
        tempt.to_csv(f"topic_trend_day_{n_topic}.csv",index=False)
    return tempt


def trend_by_day(start_date=None,end_date=None,link=None):
    # tempt = 0
    # try:
    if link == None:    
        link = str(pathlib.Path(__file__).parent.resolve()) + "/saved_data/topic_trend_day_80.csv"
    tempt = pd.read_csv(link)
    # except:
    #     raise Exception(f"can't seem to find this link :{link}\n ->{tempt}")
    # kiến cho date luôn có nghĩa - luôn đủ dạng 8 chữ số
    if start_date != None:
        number = start_date
        count = 0
        while number > 1:
            number //= 10
            count += 1
        if count == 4:
            start_date = start_date*10000
        elif count == 6:
            start_date = start_date*100
    # kiến cho date luôn có nghĩa - luôn đủ dạng 8 chữ số
    if end_date != None:
        number = end_date
        count = 0
        while number > 1:
            number //= 10

            count += 1
        if count == 4:
            end_date = end_date*10000 + 9999
        elif count == 6:
            end_date = end_date*100 + 99
        
    # kiến cho date luôn có nghĩa - luôn đủ dạng 8 chữ số, ở đây là khoảng thời gian có trong dữ liệu
    if start_date == None:
        start_date = 20070523
    if end_date == None:
        end_date = 20240608
    #if start_date > end_date:
    #    raise Exception('End date cannot be sooner than Start date')
    
    # ko cần quan tâm cái này, thầy hỏi t sẽ trả lời
    start_date = str(start_date)
    end_date = str(end_date)
    # trả về dữ liệu nằm trong khoảng thời gian được cho (từng ngày)
    return tempt.loc[:,start_date:end_date]

# tính tổng số bài của mỗi topic trong khoảng thời gian được cho
def total(start_date=None,end_date=None,link=None):
    maxtrix = trend_by_day(start_date,end_date,link)
    return maxtrix.sum(axis = 1)

def bar_visualize(vec, n_show,figsize = (20, 10)):
    # fig, ax = plt.subplots(figsize=(10, 5))

    # # Plot stacked bar chart
    # vec.iloc[:n_show].plot(kind='bar', stacked=True, ax=ax)

    # ax.set_xlabel("Date")
    # ax.set_ylabel("Number of Articles")
    # ax.legend(title='Categories')

    # return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    # Transpose the DataFrame and plot stacked bar chart
    vec.T.iloc[:, :n_show].plot(kind='bar', stacked=True, ax=ax)

    ax.set_xlabel("Categories")
    ax.set_ylabel("Number of Articles")
    ax.set_title(f"Trend of")
    ax.legend(title='Topics')

    return fig, ax

def lines_visualize(trend_data,n_show= 5, figsize = (20, 10)):

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(trend_data[:n_show])):
        x = trend_data.iloc[i].index
        y = trend_data.iloc[i].values
        ax.plot(x, y, label=f"Topic {i}")
    ax.legend(bbox_to_anchor=(0.75, 1.15), ncol=10)
    ax.set_xticklabels(trend_data.columns, rotation=90, ha='right')
    return(fig,ax)

def show_topic_total(total_topics,n_show = 5, figsize = (20,10)):

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(total_topics.index[:n_show], total_topics.values[:n_show])
    ax.set_xlabel('Topic ID')
    ax.set_ylabel('Number of Articles')
    ax.set_xticks(total_topics.index[:n_show])

    return fig, ax


def trend(start_date, end_date,groupby='day',link = None):
    # kiến cho date luôn có nghĩa - luôn đủ dạng 8 chữ số, ở đây là khoảng thời gian nằm trong dữ liệu
    if start_date == None:
        start_date = 20070523
    if end_date == None:
        end_date = 20240608

    # groupby --> muốn gộp lại để show dữ liệu ra dưới dạng từng ngày một
    #             hay theo từng tháng, từng năm (được viết ở phía dưới cùng)

    if groupby == 'day':
        return trend_by_day(start_date,end_date,link)
    
    # kiến cho date luôn có nghĩa - luôn đủ dạng 8 chữ số
    end_count=0
    start_count = 0
    if start_date != None:
        number = start_date
        while number > 1:
            number //= 10
            start_count += 1
        if start_count == 4:
            start_date = start_date*10000 + 101
        elif start_count == 6:
            start_date = start_date*100 + 1

    # kiến cho date luôn có nghĩa - luôn đủ dạng 8 chữ số
    if end_date != None:
        number = end_date
        while number > 1:
            number //= 10
            end_count += 1
        if end_count == 4:
            end_date = end_date*10000 + 9999
        elif end_count == 6:
            end_date = end_date*100 + 99

    # (trong trường hợp toàn bộ khoảng thời gian cung cấp không có dữ liệu nào)
    # cảnh báo nếu thời gian vượt ngoài khoảng có trong dữ liệu, dừng chương trình luôn
    
    if start_date > 20240608:
        raise Exception('out of bound! There is no data above there!!')
    if end_date < 20070523:
        raise Exception('out of bound! There is nothing down here!')
    
    #(trong trường hợp vẫn có một phần dữ liệu nằm trong khoảng thời gian cung cấp)
    # giúp mô hình vẫn hoạt động được bình thường bằng cách đưa khoảng thời gian đó về hoàn toàn nằm trong khoảng thời gian có dữ liệu
    if start_date < 20070523:
        start_date = 20070523
    if end_date > 20240608:
        end_date = 20240608

    #trong trường hợp ngày kết thúc nằm trước ngày bắt đầu, đảo lại vị trí của chúng (vì có thể nhỡ người dùng nhầm -> vẫn hoạt động bth)
    if start_date > end_date:
        tmp = start_date
        start_date = end_date
        end_date = tmp

    if groupby != 'year' and groupby != 'month' and groupby != 'day':
        raise Exception('grouping can only be by day,month or year')
    
    # tách nhỏ tháng, và năm ra thành từng số một để dễ xử lý
    yearstart = start_date//10000
    yearend = end_date//10000
    monthstart = int(max((start_date/100)%100,1))
    monthend = int(min((end_date/100)%100,12))


    tempts = pd.DataFrame()

    # toàn bộ phần ở dưới đây chỉ là để đảm bảo rằng, dữ liệu trả về sẽ luôn nằm trong khoảng thời gian người dùng đưa ra
    # và đúng theo kiểu  gộp dữ liệu để hiện thị mà người dùng chọn
    if groupby == 'year':
        # if within same year
        if yearstart == yearend:
            tempt = total(start_date,end_date,link)
            tempt = pd.DataFrame(tempt,columns=[f'{yearstart}'])
            return tempt


        #(start_date ---> end of the starting year)
        tempt = total(start_date,yearstart,link)
        tempt = pd.DataFrame(tempt,columns=[f'{yearstart}'])
        tempts = pd.concat([tempts,tempt], axis = 1)

        #(all the years in between start_date and end_date)
        for i in range(yearstart+1,yearend):
            tempt = total(i,i,link)
            tempt = pd.DataFrame(tempt,columns=[f'{str(i)}'])
            tempts = pd.concat([tempts,tempt], axis = 1)
        
        #(start of end year --> end_date)
        tempt = total(yearend,end_date,link)
        tempt = pd.DataFrame(tempt,columns=[f'{yearend}'])
        tempts = pd.concat([tempts,tempt], axis = 1)
        return tempts
    
    if groupby == 'month':


        #if within the same year
        if yearstart == yearend:
            for month in range(monthstart,monthend+1):
                _month = yearstart*10000+month*101
                month_ = yearend*10000+month*100+32
                tempt = total(max(_month,start_date),min(month_,end_date),link)

                tempt = pd.DataFrame(tempt,columns=[f'{str(_month//100)}'])
                tempts = pd.concat([tempts,tempt], axis = 1)
            return tempts
        
        #(start_date  -->  12)
        for month in range(monthstart,12+1):
            _month = yearstart*10000 + month*100
            month_ = yearstart*100 + month
            tempt = total(max(start_date,_month),month_,link)


            tempt = pd.DataFrame(tempt,columns=[f'{str(month_)}'])
            tempts = pd.concat([tempts,tempt], axis = 1)   
        
        # all the years in between start_date and end_date
        for year in range(yearstart+1,yearend):
            for month in range(1,12 + 1):
                tempt = total(year*100+month,year*100+month,link)
                
                tempt = pd.DataFrame(tempt,columns=[f'{str(year*100+month)}'])
                tempts = pd.concat([tempts,tempt], axis = 1)

        #(1 - > end_date)
        for month in range(1,monthend+1):
            _month = yearend*100 + month
            month_ = yearend*10000 + month*100 + 32
            tempt = total(_month,min(end_date,month_),link)

            tempt = pd.DataFrame(tempt,columns=[f'{str(_month)}'])
            tempts = pd.concat([tempts,tempt], axis = 1)   
        
        return tempts


def strip_spaces(text):
 
  #Loại bỏ khoảng trắng ở đầu và cuối chuỗi.

  return text.strip()

def convert_dash(text):
    return text.replace('\\','/')

def read_text_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        return file_path
    except Exception as e:
        return f"An error occurred: {e}"

def read_doc(text):
        link = convert_dash(text)
        text = read_text_from_file(link)
        return text



def doc_info(content,lda_address):

    lda_model = LdaModel.load(lda_address)
    dictionary = lda_model.id2word

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    content = content.lower()
    content = word_tokenize(content)
    content = [lemmatizer.lemmatize(word) for word in content if word not in stop_words]
    bow = dictionary.doc2bow(content)
        

    doc_topics = lda_model.get_document_topics(bow)
    #topics = lda_model.show_topic()
    return doc_topics
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch





def show_topic_radar(total,n_show = 5,figsize = (20, 10)):


    total_articles = total.sum()
    topics = [(topic_id, count / total_articles) for topic_id, count in total.items()]

    values = [probability for topic_id, probability in topics[:n_show]]
    categories = [f'Topic {topic_id}' for topic_id, probability in topics[:n_show]]

    num_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig,ax = plt.subplots(figsize=figsize)
    
    ax = fig.add_subplot(111, polar=True)

  

    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1], categories)

    return fig, ax

def doc_topic_radar(total,n_show = 5,figsize = (10, 10)):

    total = total.values

    values = [probability for topic_id, probability in total[:n_show]]
    categories = [f'Topic {int(topic_id)}' for topic_id, probability in total[:n_show]]

    num_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig,ax = plt.subplots(figsize=figsize)
    
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1], categories)


    return fig, ax


def generate_word_cloud(topic_id,link):
    lda_model = LdaModel.load(link)

  
    topic_words = lda_model.show_topic(topic_id, topn=20)
    topic_dict = {word: prob for word, prob in topic_words}
    

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_dict)
    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    ax.set_title(f'Topic {topic_id}')
    return fig, ax


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt






#def main():
st.title("Trend Analysis and Article Search")

if 'search_engine' not in st.session_state:
    placeholder = st.empty()
    placeholder.write('Loading in data.........')
    st.session_state.search_engine = blind_search_engine()
    placeholder.write('Finished Loading in data')

# Access the search engine from the session state
search_engine = st.session_state.search_engine

#keyword = st.text_input("Enter a keyword:")

st.sidebar.header("Input Parameters")
analysis_type = st.sidebar.radio("Select what you want to do", ("Trend Analysis", "Article Search","Document Analysis"))

if analysis_type == "Trend Analysis":
    start_date = st.sidebar.text_input("Start Date (YYYYMMDD)", value="20070523")
    end_date = st.sidebar.text_input("End Date (YYYYMMDD)", value="20240608")
    n_show = st.sidebar.text_input("Number of Top Topic to show", value="5")
    groupby = st.sidebar.selectbox("Group By", ["day", "month", "year"])
    link = st.sidebar.text_input("CSV Link (optional)",value = '')

    if strip_spaces(link) == '':
        link = None

    try:
        if n_show == '':
            n_show = 5
        n_show = int(n_show)
    except:
        raise Exception('value of number of topic showing is not appropriate')

    if st.sidebar.button("Show Trend"):
        try:
            start_date = int(start_date)
            end_date = int(end_date)
            trend_data = trend(start_date, end_date, groupby, link)
            st.write(trend_data)

            st.subheader("Bar Visualization")
            fig,ax = bar_visualize(trend_data,n_show)
            st.pyplot(fig)

            st.subheader("Line Visualization")
            fig,ax = lines_visualize(trend_data,n_show)
            st.pyplot(fig)
            
            print(total(start_date,end_date,link))
            top = total(start_date,end_date,link).sort_values(ascending=False)

            st.subheader(f"Total topics from {start_date} to {end_date}")
            fig,ax = show_topic_total(top,n_show)
            st.pyplot(fig)

            st.subheader(f"Rader Visualize of topics from {start_date} to {end_date}")
            fig,ax = show_topic_radar(top,n_show)
            st.pyplot(fig)

            

        except Exception as e:
            st.error(f"Error: {e}")

elif analysis_type == "Article Search":
    keyword = st.sidebar.text_input("Enter a keyword:")
    
    if st.sidebar.button("Search"):
        keyword = strip_spaces(keyword)
        if keyword:
            try:
                results,probs = search_engine.search(keyword)
            except:
                raise Exception("Found Nothing")
            
            if not results.empty:
                i = 0
                st.subheader(f"Results for '{keyword}':")
                for index, row in results.iterrows():
                    st.markdown(f"Correlation Confidence: {round(probs[i]*100,1)}%")
                    i = i+ 1
                    st.markdown(f"ID: {row['id']}")
                    st.markdown(f"Title: {row['title']}")
                    st.markdown(f"DOI: {row['doi']}")
                    st.markdown(f"Abstract: {row['abstract'][:200]}...")  # Displaying first 200 characters
                    with st.expander("Full Abstract"):
                        st.write(row['abstract'])
                    st.markdown("---")
            else:
                st.write("No results found.")
        else:
            st.write("Please enter a keyword.")

elif analysis_type == "Document Analysis":
    doc = st.sidebar.text_input("Enter a Document or Document's Path:", value="C:/Users\japan\OneDrive\Desktop\super_start.txt")
    n_show = st.sidebar.text_input("Number of top relevant topics showing:", value = 5)
    try:
        n_show = int(n_show)
    except TypeError:
        raise Exception('Number of top relevant topics')
    

    link = st.sidebar.text_input("LdaModel Address (Optional):")
    link = strip_spaces(link)
    if link == '':
        link = str(pathlib.Path(__file__).parent.resolve()) + '/saved_data/lda_model_80.model'
    link = convert_dash(link)

    if st.sidebar.button("Analyze"):

        if doc:
            content = read_doc(doc)
            doc_topics = doc_info(content,link)
            topic = sorted(doc_topics, key=lambda x: x[1], reverse=True)[:n_show]
            doc_topics = pd.DataFrame(topic)
            
            topic_ids = [topic[index][0] for index in range(len(topic))]
            top_probs = [f"{round(topic[index][1]*100,2)}%" for index in range(len(topic))]

            topic_ids = pd.DataFrame(topic_ids, columns=['Topic ID'])
            top_probs = pd.DataFrame(top_probs, columns=['percent'])
            topics = pd.concat([topic_ids,top_probs],axis = 1)
            st.write(topics)

            st.subheader("Radar Chart for the Document' topics")
            fig, ax = doc_topic_radar(doc_topics,n_show = len(doc_topics))
            st.pyplot(fig)

            st.subheader(f"Word Cloud of Topic ID: {topic[0][0]}")
            fig, ax = generate_word_cloud(topic[0][0],link)
            st.pyplot(fig)
        else:
            st.write("Please enter Document or Document's Path")
    


