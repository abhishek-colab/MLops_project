import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import  StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def read_process_df(path, spark):
    df = spark.read.format('parquet').load(path)
    df = df.select('VendorID','lpep_pickup_datetime','lpep_dropoff_datetime','PULocationID','DOLocationID','trip_distance')
    df = df.withColumn('duration',\
        round((col('lpep_dropoff_datetime')-col('lpep_pickup_datetime'))\
        .cast("long")/60,2))
    df = df.filter(col('duration')>=0.05).filter(col('duration')<=82)
    df = df.withColumn('PU_DO',concat(col('PULocationID'),lit('_'),col('DOLocationID')))
    df = df.withColumn('pu_hour',hour(col('lpep_pickup_datetime')))
    df = df.withColumn('pu_weekday',dayofweek(col('lpep_pickup_datetime')))
    
    df = df.select('VendorID','pu_hour','pu_weekday','PU_DO', 'trip_distance','duration')
    # y = df.select('')

    print(df.count(), len(df.columns))
    return df

def prepare_data(df_processed,categorical_cols,indexer_final=None,encoder_final=None, is_test=True):
    indexers = [StringIndexer(inputCol=col,outputCol=col+'_index').fit(df_processed) for col in categorical_cols]
    if is_test:
        df_processed = df_processed.dropna(subset='duration')
        [indexer.setHandleInvalid("keep") for indexer in indexers]
    indexer_pipeline = Pipeline(stages=indexers)
    if indexer_final==None:
        indexer_final = indexer_pipeline.fit(df_processed)
    
    indexed_df = indexer_final.transform(df_processed)

    encoder = [OneHotEncoder(inputCol=col+'_index',outputCol=col+'_onehot') for col in categorical_cols]
    encoder_pipeline = Pipeline(stages = encoder)
    if encoder_final==None:
        encoder_final = encoder_pipeline.fit(indexed_df)
    encoded_df = encoder_final.transform(indexed_df)
    
    return encoded_df, indexer_final, encoder_final
    

def train_model(encoded_df,feature_cols,label_col,regressor = LinearRegression):
    assembler = VectorAssembler(inputCols = feature_cols, outputCol = 'features')
    regressor = regressor(featuresCol = 'features', labelCol= label_col )
    pipeline = Pipeline(stages = [assembler,regressor])
    model = pipeline.fit(encoded_df)
    return model

def evaluate_model(model,encoded_df,label_col,metric='rmse'):
    predictions = model.transform(encoded_df)
    evaluator = RegressionEvaluator(labelCol=label_col,predictionCol='prediction',metricName=metric)
    out = evaluator.evaluate(predictions)
    print(f"{metric}  : {out}")
    
    return out
