from pyspark.sql.functions import col, when

def clean_initial_columns(df):
    """Supprime les colonnes inutiles dès le début (utilisé dans le train)"""
    drop_cols = ['EmployeeNumber','Over18','StandardHours','EmployeeCount','MonthlyRate','PercentSalaryHike']
    # On ne supprime que si les colonnes existent
    existing_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(*existing_cols)

def apply_transformations(df):
    """
    Applique toutes les règles métiers (booléens, calculs)
    Cette fonction est utilisée à la fois par le TRAIN et le STREAM.
    """
    
    # 1. Encodage binaire simple
    # Note: On vérifie si la colonne existe pour éviter des erreurs si le format change
    if "OverTime" in df.columns:
        df = df.withColumn("OverTime", when(col("OverTime")=="Yes", 1).otherwise(0))
    if "Gender" in df.columns:
        df = df.withColumn("Gender", when(col("Gender")=="Female", 1).otherwise(0))

    # 2. Calcul Satisfaction
    # On suppose que les colonnes nécessaires sont présentes
    required_satisfaction_cols = ["EnvironmentSatisfaction", "JobInvolvement", 
                                  "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
    
    # Vérification simple que toutes les colonnes sont là avant de calculer
    if all(c in df.columns for c in required_satisfaction_cols):
        df = df.withColumn("Total_Satisfaction", (
            col("EnvironmentSatisfaction") +
            col("JobInvolvement") +
            col("JobSatisfaction") +
            col("RelationshipSatisfaction") +
            col("WorkLifeBalance"))/5
        )
        df = df.withColumn("Total_Satisfaction_bool", when(col("Total_Satisfaction")>=2.8, 1).otherwise(0))
        # On supprime les colonnes sources pour alléger (comme dans votre script train original)
        df = df.drop(*required_satisfaction_cols, "Total_Satisfaction")

    # 3. Règles métiers (Création booléen + Suppression colonne source)
    # Dictionnaire: { "Nom_Col_Source": (Seuil, "Nom_Col_Cible", operateur) }
    # Pour garder votre logique exacte, on le fait explicitement :

    if "Age" in df.columns:
        df = df.withColumn("Age_bool", when(col("Age")<35, 1).otherwise(0)).drop("Age")
        
    if "DailyRate" in df.columns:
        df = df.withColumn("DailyRate_bool", when(col("DailyRate")<800, 1).otherwise(0)).drop("DailyRate")
        
    if "Department" in df.columns:
        df = df.withColumn("Department_bool", when(col("Department")=="Research & Development", 1).otherwise(0)).drop("Department")

    if "DistanceFromHome" in df.columns:
        df = df.withColumn("DistanceFromHome_bool", when(col("DistanceFromHome")>10, 1).otherwise(0)).drop("DistanceFromHome")
        
    if "JobRole" in df.columns:
        df = df.withColumn("JobRole_bool", when(col("JobRole")=="Laboratory Technician", 1).otherwise(0)).drop("JobRole")
        
    if "HourlyRate" in df.columns:
        df = df.withColumn("HourlyRate_bool", when(col("HourlyRate")<65, 1).otherwise(0)).drop("HourlyRate")
        
    if "MonthlyIncome" in df.columns:
        df = df.withColumn("MonthlyIncome_bool", when(col("MonthlyIncome")<4000, 1).otherwise(0)).drop("MonthlyIncome")
        
    if "NumCompaniesWorked" in df.columns:
        df = df.withColumn("NumCompaniesWorked_bool", when(col("NumCompaniesWorked")>3, 1).otherwise(0)).drop("NumCompaniesWorked")
        
    if "TotalWorkingYears" in df.columns:
        df = df.withColumn("TotalWorkingYears_bool", when(col("TotalWorkingYears")<8, 1).otherwise(0)).drop("TotalWorkingYears")
        
    if "YearsAtCompany" in df.columns:
        df = df.withColumn("YearsAtCompany_bool", when(col("YearsAtCompany")<3, 1).otherwise(0)).drop("YearsAtCompany")
        
    if "YearsInCurrentRole" in df.columns:
        df = df.withColumn("YearsInCurrentRole_bool", when(col("YearsInCurrentRole")<3, 1).otherwise(0)).drop("YearsInCurrentRole")
        
    if "YearsSinceLastPromotion" in df.columns:
        df = df.withColumn("YearsSinceLastPromotion_bool", when(col("YearsSinceLastPromotion")<1, 1).otherwise(0)).drop("YearsSinceLastPromotion")
        
    if "YearsWithCurrManager" in df.columns:
        df = df.withColumn("YearsWithCurrManager_bool", when(col("YearsWithCurrManager")<1, 1).otherwise(0)).drop("YearsWithCurrManager")

    return df
