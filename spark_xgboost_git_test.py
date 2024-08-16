# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline

# Sample data
data = [
    (1, 1.0, 0.1, 0.2, 0.3),
    (0, 0.2, 0.1, 0.4, 0.4),
    (1, 0.3, 0.2, 0.5, 0.5),
    (0, 0.4, 0.3, 0.6, 0.6),
    (1, 0.5, 0.4, 0.7, 0.7),
]

# Create DataFrame
columns = ["label", "feature1", "feature2", "feature3", "feature4"]
df = spark.createDataFrame(data, columns)

# Assemble features into a single vector column
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3", "feature4"],
    outputCol="features"
)

# Define the XGBoost classifier
xgb_classifier = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    prediction_col="prediction",
    max_depth=3,
    eta=0.1,
    num_round=50
)

# Create a pipeline
pipeline = Pipeline(stages=[assembler, xgb_classifier])

# Train the model
model = pipeline.fit(df)

# Make predictions
predictions = model.transform(df)

# Show predictions
display(predictions.select("label", "features", "prediction"))

# COMMAND ----------

print(model.stages)

# COMMAND ----------


