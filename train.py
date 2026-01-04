from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# IMPORT DE NOTRE MODULE
from preprocessor import apply_transformations, clean_initial_columns

spark = SparkSession.builder.appName("EmployeeAttrition_Train").getOrCreate()

# 1. Lecture
df = spark.read.csv("datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=True, inferSchema=True)

# 2. Nettoyage initial (Colonnes inutiles)
df = clean_initial_columns(df)

# 3. Traitement de la Target (Attrition) - Spécifique au Train
df = df.withColumn("Attrition", when(col("Attrition")=="Yes", 1).otherwise(0))

# 4. APPEL DU MODULE PARTAGÉ (Transformations Booléennes)
df = apply_transformations(df)

# ======================
# Construction du Pipeline ML
# ======================
categorical_cols = ['BusinessTravel','Education','EducationField','MaritalStatus','StockOptionLevel','TrainingTimesLastYear']

stages = []
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_Index", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=[col_name+"_Index"], outputCols=[col_name+"_OHE"])
    stages += [indexer, encoder]

# On récupère les colonnes numériques automatiquement (celles qui restent après transformations)
numeric_cols = [c for c in df.columns if c not in categorical_cols + ["Attrition"]]
assembler_inputs = [c+"_OHE" for c in categorical_cols] + numeric_cols

assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
stages.append(assembler)

lr = LogisticRegression(featuresCol="features", labelCol="Attrition", maxIter=10)
stages.append(lr)

pipeline = Pipeline(stages=stages)

# ======================
# Split, Fit & Save
# ======================
train_df, test_df = df.randomSplit([0.8,0.2], seed=42)
model = pipeline.fit(train_df)

model_path = "/home/ubuntu/Employee-Attrition-Prediction/HRAttrition_Model"
model.write().overwrite().save(model_path)
print(f"Modèle sauvegardé dans {model_path}")

# Petit test rapide
model.transform(test_df).select("Attrition", "prediction").show(5)
