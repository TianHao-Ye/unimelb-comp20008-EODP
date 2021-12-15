# unimelb-comp20008-EODP
Introduction
This project is designed to give you practice at solving some practical data science tasks. You
will need to implement and evaluate a data linkage system and a classication system using
sound data science principles. We intentionally do not prescribe the exact steps required as
a key part of the assessment is to evaluate your ability to exercise your own judgement in
deciding how to solve these practical data science problems, rather than simply implement-
ing a pre-dened algorithm. This is a big part of any real-world data science task, as there
are usually many approaches that can be taken and many irregularities in the data that are
impossible to exhaustively predene in a task description. A data scientist needs to be able
to select the best approach for resolving these issues and justify it in order for their results
to be convincing.

For this project, you will perform a data linkage on two real-world datasets (Part 1) and
explore dierent classication algorithms (Part 2).

Part 1 - Data Linkage (12 marks)
Abt and Buy both have product databases. In order to perform joint market research they
need to link the same products in their databases. Therefore the research team has manually
linked a subset of the data in order to build a linkage algorithm to automate the remaining
items to be linked. This manually linked data is what you will base your work on in this
assignment. However the real dataset is unknown to you, as it would be reality and this
unknown data is what you will be assessed on.

Nave data linkage without blocking (4 marks)
For this part, data linkage without blocking is performed on two smaller data sets:
abt small.csv and buy small.csv.
Task - 1A: Using abt small.csv and buy small.csv, implement the linkage between the
two data sets.
Your code for this question is to be contained in a single Python le called task1a.py and
produce a single csv le task1a.csv containing the following two column headings:
idAbt,idBuy
Each row in the datale must contain a pair of matched products. For example, if your
algorithm only matched product 10102 from the Abt dataset with product
203897877 from the Buy dataset your output task1a.csv would be as follows:
idAbt, idBuy
10102,203897877
The performance is evaluated in terms of recall and precision and the marks in this section
will be awarded based on the two measures of your algorithm.
recall = tp=(tp + fn)
precision = tp=(tp + fp)
where tp (true-positive) is the number of true positive pairs, fp the number of false positive
pairs, tn the number of true negatives, and fn the number of false negative pairs.
Note: The python package textdistance implements many similarity functions for strings
(https://pypi.org/project/textdistance/). You can use this package for the similarity
calculations for strings. You may also choose to implement your own similarity functions.

Blocking for ecient data linkage (4 marks)
Blocking is a method to reduce the computational cost for record linkage.
Task - 1B: Implement a blocking method for the linkage of the abt.csv and buy.csv data
sets.
Your code is be contained in a single Python le called task1b.py and must produce two
csv les abt blocks.csv and buy blocks.csv, each containing the following two column
headings:
block_key, product_id
The product id eld corresponds to the idAbt and idBuy of the abt.csv and buy.csv les
respectively. Each row in the output les matches a product to a block. For example, if your
algorithm placed product 10102 from the Abt dataset in blocks with block keys x & y, your
abt blocks.csv would be as follows:
block_key, product_id
x,10102
y,10102
A block is uniquely identied by the block key. The same block key in the two block-les
(abt blocks.csv and buy blocks.csv) indicates that the corresponding products co-occur
in the same block.
For example, if your algorithm placed the Abt product 10102 in block x and placed Buy
product 203897877 in block x, your abt blocks.csv and buy blocks.csv would be as follows
respectively:
abt_blocks.csv:
block_key, product_id
x,10102
buy_blocks.csv:
block_key, product_id
x,203897877
The two products co-occur in the same block x.
To measure the quality of blocking, we assume that when comparing a pair of records, the
pair are always 100% similar and are a match. A pair of records are categorised as follows:
 a record-pair is a true positive if the pair are found in the ground truth set and also the
pair co-occur in the same block.
 a record-pair is a false positive if the pair co-occur in some block but are not found in
the ground truth set.
 a record-pair is a false negative if the pair do not co-occur in any block but are found
in the ground truth set

 a record-pair is a true negative if the pair do not co-occur in any block and are also not
found in the ground truth set.
Then, the quality of blocking can be evaluated using the following two measures:
PC (pair completeness) = tp=(tp + fn)
RR (reduction ratio) = 1 􀀀 (tp + fp)=n
where n is the total number of all possible record pairs from the two data sets
(n = fp + fn + tp + tn).
The marks in this section will be awarded based on the pair completeness and reduction ratio
of your blocking algorithm.
Note: The time taken to produce your blocking implementation must be linear in the number
of items in the dataset. This means that you cannot, for example, compare each product to
every other product in the dataset in order to allocate it to a block. Implementations that
do so will be severely penalised.
