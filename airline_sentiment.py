import pandas as pd
# reading data from csv file
df = pd.read_csv('~/Downloads/Tweets.csv')
df.head(5)
df.groupby('airline').count()
df.drop('negativereason',1,inplace=True)
df.drop('negativereason_confidence',1,inplace=True)
df.drop('name',1,inplace=True)
df.drop('negativereason_gold',1,inplace=True)
df.drop('retweet_count',1,inplace=True)
df.drop('tweet_coord',1,inplace=True)
df.drop('tweet_created',1,inplace=True)
df.drop('user_timezone',1,inplace=True)
df.drop('tweet_location',1,inplace=True)
df.head(3)
df['text'].head(10)

# cleaning the data
import re
import string
def clean_data(input):
    arrays=[] 
    for processed in input.split():
        processed=processed.rstrip()
        processed=processed.replace('\"','')
        processed=processed.strip('\" ')
        processed=processed.strip('“').strip('”')
        processed=processed.replace('\'',"")
        processed=processed.replace('\.',"") 
        processed=processed.replace('#',"") 
        processed=processed.replace('\;',"")
        processed=processed.replace('\:',"")
        processed=processed.replace('\?',"")
        processed=processed.replace('!',"")
        processed=re.sub('@[^\s]+','',processed)
        processed=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',processed)
        arrays.append(processed)
    return ' '.join(arrays) 
df['text_processed'] = df['text'].apply(lambda x: clean_data(x))
df.head(5)

# removing stop words
s={'ourselves','&lt;3','&gt;','l&amp;f',':-D',':)','lax---&gt;',':(',':-d','&amp;','hers','as','between','i','you','me','my','yourself', 'they','but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
def remove_stopwords(input):
    new_arrays=[] 
    for i in input.lower().split():
        if i not in s:
            if len(i) >= 3:
                new_arrays.append(i)
    return ' '.join(new_arrays)
            
df['cleansed_data'] = df['text_processed'].apply(lambda x: remove_stopwords(x))
df.head(5)

# removing non-ascii words from the text
def strip_non_ascii(input):
    new_arrays=[] 
    for i in input.split():
        stripped = (c for c in i if 0 < ord(c) < 127)
        i=''.join(stripped)
        new_arrays.append(i)
    return ', '.join(new_arrays)
 
df['cleansed_data'] = df['cleansed_data'].apply(lambda x: strip_non_ascii(x))
df.head(5)

#taking data with confidence value 1
new_df=df[df['airline_sentiment_confidence']==1]

# seperating sentiment and cleansed text for further processing
new_df.to_csv('result.csv',index=False)
f=open('result.csv', 'r')
inpTweets = csv.reader(f)
tweets = []
for row in inpTweets:
    sentiment = row[1]
    tweet = row[6]
    tweets.append((sentiment,tweet))
tweets = numpy.delete(tweets, (0), axis=0)

my_df = pd.DataFrame(tweets)
my_df.shape

# saving dataframe to csv file
my_df.to_csv('clean.csv', header=None, index=None, sep=',', mode='w')

# reading data from csv file
pandas_df = pd.read_csv("~/Downloads/clean.csv", names=['label','text'])
# Removing empty texts
pandas_df = pandas_df.dropna(axis=0)
# Reducing data to 100 samples
pandas_df = pandas_df[:3000]

content = []
for index, row in pandas_df.iterrows():
  k = int(index), row['text'].split(','), int(row['label'])
  content.append(tuple(k))

df = spark.createDataFrame(content, ["id", "text", "label"])

# Count Vectorizer to vectorize text
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="text", outputCol="features")
model = cv.fit(df)
result = model.transform(df)

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector
labelPoint = result.select("label","features").rdd.map(lambda row: LabeledPoint(row[0], DenseVector(row[1])))

# Split data approximately into training (60%) and test (40%)
training, test = labelPoint.randomSplit([0.7, 0.3])

from pyspark.mllib.classification import NaiveBayes
# Train a naive Bayes model.
model = NaiveBayes.train(training, 1.0)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
print('model accuracy {}'.format(accuracy))