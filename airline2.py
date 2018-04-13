data_rdd = sc.textFile("new_tfidf.csv")
parts_rdd = data_rdd.map(lambda l: l.split("\t"))
# Filter bad rows out
garantee_col_rdd = parts_rdd.filter(lambda l: len(l) == 2)
typed_rdd = garantee_col_rdd.map(lambda p: (p[0], float(p[1])))
typed_rdd.take(4)
# Create DataFrame
data_df = spark.createDataFrame(typed_rdd, ["text","label"])
data_set =data_df.select(data_df['label'],data_df['text'])
#splitting data to train and test
training_df,test_df= data_set.randomSplit([0.7, 0.3])
training_df.head(5)

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
idf = IDF(minDocFreq=3, inputCol="features", outputCol="idf")
nb = NaiveBayes()
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])

paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 1.0]).build()


cv = CrossValidator(estimator=pipeline, 
                    estimatorParamMaps=paramGrid, 
                    evaluator=MulticlassClassificationEvaluator(), 
                    numFolds=4)

cvModel = cv.fit(training_df)

result = cvModel.transform(test_df)
prediction_df = result.select("text", "label", "prediction")

datasci_df = prediction_df.filter(prediction_df['label']==0.0)
datasci_df.show(truncate=False)

ao_df = prediction_df.filter(prediction_df['label']==1.0)
ao_df.show(truncate=False)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(result, {evaluator.metricName: "accuracy"})
