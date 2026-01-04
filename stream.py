from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split
from pyspark.ml import PipelineModel

# IMPORT DE NOTRE MODULE
from preprocessor import apply_transformations

spark = SparkSession.builder.appName("EmployeeAttrition_RealTime").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 1. Chargement du Modèle
model_path = "/home/ubuntu/Employee-Attrition-Prediction/HRAttrition_Model"
model = PipelineModel.load(model_path)

# 2. Lecture Socket
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "16.171.250.66") \
    .option("port", 9999) \
    .load()

# 3. Parsing (Définition du schéma d'entrée)
columns = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked", "OverTime",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

df = raw_stream.select(
    *[split(col("value"), ",").getItem(i).alias(columns[i]) for i in range(len(columns))]
)

# 4. Cast des types (Int)
int_cols = [
    "Age","DailyRate","DistanceFromHome","Education","EnvironmentSatisfaction",
    "HourlyRate","JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome",
    "NumCompaniesWorked","PerformanceRating","RelationshipSatisfaction",
    "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
    "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
    "YearsSinceLastPromotion","YearsWithCurrManager"
]

for c in int_cols:
    df = df.withColumn(c, col(c).cast("int"))

# 5. APPEL DU MODULE PARTAGÉ (Transformations)
# C'est ici que la magie opère : une seule ligne remplace 30 lignes de code dupliqué
df = apply_transformations(df)

# 6. Prédiction & Sortie
predictions = model.transform(df)

query = predictions.select("prediction", "probability") \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
