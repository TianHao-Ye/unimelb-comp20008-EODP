import pandas as pd
import textdistance
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

#load the data
abt_s=pd.read_csv('abt_small.csv',encoding = 'ISO-8859-1')
buy_s=pd.read_csv('buy_small.csv',encoding = 'ISO-8859-1')
df_abt = pd.DataFrame(abt_s)
df_buy = pd.DataFrame(buy_s)

sorensen = textdistance.sorensen
levenshtein = textdistance.levenshtein
lemmatizer = WordNetLemmatizer()
porterStemmer = PorterStemmer()
stopWords = set(stopwords.words('english'))

#record all mathced pairs
all_pairs = []

#natural language processing
def nlp_string(np_list):
    np_list = np_list.lower()
    #noise removal
    np_list = re.sub(r'[^a-zA-Z0-9.-/ ]',r'', np_list)
    np_list = re.sub(r'( - |/ )',r' ',np_list)
    #tokenisation
    wordList = nltk.word_tokenize(np_list)
    #lemmatsation
    lemmatizedList = []
    for word in wordList:
        lemmatizedWord = lemmatizer.lemmatize(word)
        lemmatizedList.append(lemmatizedWord)
    #stop words removal
    filteredList = [w for w in lemmatizedList if not w in stopWords]
    return filteredList

#dealing with missing data
df_buy = df_buy.fillna("NAN_VALUE")
df_abt = df_abt.fillna("NAN_VALUE")

#processing buy_records
buy_records = []
for index, row_buy in df_buy.iterrows():
    buy_record = []
    id_buy = str(row_buy["idBuy"])
    name_buy = nlp_string(str(row_buy["name"]))
    description_buy = nlp_string(str(row_buy["description"]))
    price_buy = str(row_buy["price"])
    
    buy_record.append(id_buy)
    buy_record.append(name_buy)
    buy_record.append(description_buy)
    buy_record.append(price_buy)
    buy_records.append(buy_record)

#iterate abt
for index, row_abt in df_abt.iterrows():
    is_d_valid_abt = True
    is_p_valid_abt = True
    new_pair = []
    largest_similarity = 0
    largest_buy_id = 0
    
    #process data in abt
    
    id_abt = row_abt[0]
    name_abt = nlp_string(str(row_abt["name"]))
    description_abt = nlp_string(str(row_abt["description"]))
    price_abt = str(row_abt["price"])
    if(row_abt["price"] == "NAN_VALUE"):
        is_p_valid_abt = False
            
    for buy_record in buy_records:
        is_d_valid = True
        is_p_valid = True
        
        #calculate the similarities of feature pairs
        s_name = sorensen.normalized_similarity(name_abt , buy_record[1])
        if(buy_record[2] != "NAN_VALUE" and is_d_valid_abt==True):
            s_description = sorensen.normalized_similarity(description_abt , buy_record[2])
        else:
            is_d_valid = False
         
        if(buy_record[3] != "NAN_VALUE" and is_p_valid_abt==True):
            s_price = levenshtein.normalized_similarity(price_abt , buy_record[3])
        else:
            is_p_valid = False
        
        #calculate the weighted avg of similarity in different circumstances
        if(is_d_valid and is_p_valid):
            avg_similarity = 0.85*s_name+0.1*s_description+0.05*s_price
        elif(is_d_valid and is_p_valid==False):
            avg_similarity = 0.8*s_name+0.2*s_description
        elif(is_d_valid==False and is_p_valid):
            avg_similarity = 0.9*s_name+0.1*s_price
        else:
            avg_similarity = s_name
                    
        #find the item with the largest similarity
        if (avg_similarity > largest_similarity):
            largest_similarity = avg_similarity
            largest_buy_id = buy_record[0]
    
    if (largest_similarity > 0.4):
        new_pair.append(id_abt)
        new_pair.append(largest_buy_id)
        all_pairs.append(new_pair)
        
column_names = ["idAbt", "idBuy"]
pair_data = pd.DataFrame(all_pairs, columns = column_names)
pair_data.to_csv('task1a.csv', index= False)