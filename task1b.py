import pandas as pd
import re
import nltk

#load the data
abt=pd.read_csv('abt.csv',encoding = 'ISO-8859-1')
buy=pd.read_csv('buy.csv',encoding = 'ISO-8859-1')
df_abt = pd.DataFrame(abt)
df_buy = pd.DataFrame(buy)

#dealing with missing data
df_buy = df_buy.fillna("NAN_VALUE")
df_abt = df_abt.fillna("NAN_VALUE")

def nlp(string):
    string = string.lower()
    string = re.sub(r'(- | -| - )', r' - ', string)
    string = row_name = re.sub(r'[^a-zA-Z0-9_-]+', r' ', string)
    return string

#recording all block keys
blocks = {}

#building blocks
for index, row_abt in df_abt.iterrows():
    #pre-processing abt name 
    row_name = nlp(row_abt["name"])
    row_price = row_abt["price"]
    #building blocks
    block_element = []
    block_a = re.search(r'^[^ ]+', row_name).group(0)
    block_b = re.search(r'[^ ]* -', row_name).group(0)
    block_b = re.search(r'^\w+', block_b).group(0)
    block_c = ''
    
    block_element.append(block_a.strip())
    block_element.append(block_b.strip())
    if(row_price != "NAN_VALUE"):
        block_c = block_c + row_price
        block_element.append(block_c.strip())
    block_key = ' '.join(key for key in block_element).strip()
    #record built block
    if block_key not in blocks:
        blocks[block_key] = block_element
    #add a new column manufacturer for convenience of blocking
    df_abt.loc[index,'manufacturer'] = block_a.strip()

#blocking algorithm
def blocking(df):
    block = {}
    block["non-pair"] = set()
    
    for index, row in df.iterrows():
        is_allocated = False
        row_name = row["name"].lower()
        row_description = row["description"].lower()
        row_manufacturer = row["manufacturer"].lower()
        row_price = row["price"]
        
        #round 1 blocking
        for block_key in blocks:
            block_key_elements = blocks[block_key]
            element_a = block_key_elements[0]
            element_b = block_key_elements[1]
        
            #product name and device use both matched, add into the block
            if ((re.search("%s" % element_a, row_name) != None and re.search("%s" % element_b, row_name) != None) or
               (re.search("%s" % element_a, row_description) != None and re.search("%s" % element_b, row_description) != None)):
                is_allocated = True
                if(block_key in block):
                    block[block_key].add(row[0])
                else:
                    block[block_key] = set()
                    block[block_key].add(row[0])
                
            #price matched , add into the block
            if(len(block_key_elements)>2):
                element_c = block_key_elements[2]
                if(re.search("%s" % element_c, row_price) != None):
                    is_allocated = True
                    if(block_key in block):
                        block[block_key].add(row[0])
                    else:
                        block[block_key] = set()
                        block[block_key].add(row[0])
        
        #round 2 blocking, reducing blocking threshold
        if(is_allocated == False):
            for block_key in blocks:
                block_key_elements = blocks[block_key]
                element_a = block_key_elements[0]
                element_b = block_key_elements[1]
                if (re.search("%s" % element_a, row_name) != None or re.search("%s" % element_b, row_name) != None or
                    re.search("%s" % element_a, row_description) != None or re.search("%s" % element_b, row_description) != None or
                    re.search("%s" % element_a, row_manufacturer) != None):
                    is_allocated = True
                    if(block_key in block):
                        block[block_key].add(row[0])
                    else:
                        block[block_key] = set()
                        block[block_key].add(row[0])
                #if price is same as the one in block, add into the block
                if(len(block_key_elements) > 2):
                    element_c = block_key_elements[2]
                    if(re.search("%s" % element_c, row_price) != None):
                        is_allocated = True
                        if(block_key in block):
                            block[block_key].add(row[0])
                        else:
                            block[block_key] = set()
                            block[block_key].add(row[0])
        #round 3 blocking
        if(is_allocated == False):
            block["non-pair"].add(row[0])
    return block

#processing dataframes with the blocking algorithm
block_abt = blocking(df_abt)
block_buy = blocking(df_buy)

def produce_pairs(block):
    pairs = []
    for block_key in block:
        records = block[block_key]
        for record in records:
            new_pair = []
            new_pair.append(block_key)
            new_pair.append(record)
            pairs.append(new_pair)
    return pairs

abt_pairs = produce_pairs(block_abt)
buy_pairs = produce_pairs(block_buy)
        
column_names = ["block_key", "product_id"]
abt_blocks = pd.DataFrame(abt_pairs, columns = column_names)
abt_blocks.to_csv('abt_blocks.csv', index= False)

buy_blocks = pd.DataFrame(buy_pairs, columns = column_names)
buy_blocks.to_csv('buy_blocks.csv', index= False)