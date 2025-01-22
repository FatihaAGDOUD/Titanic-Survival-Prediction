from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialisation de Spark
spark = SparkSession.builder.appName("Titanic_Classification")  \
    .config("spark.master","spark://spark-master:7077")\
    .getOrCreate()

# Charger les données depuis HDFS (ou une autre source)
data_path = "hdfs://namenode:9000/user/data/titanic.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Afficher les colonnes pour comprendre la structure des données
df.printSchema()

# Filtrer et sélectionner les colonnes utiles
df = df.select(
    col("Survived").alias("label"),  # Étiquette : survie (0 ou 1)
    "Pclass",     # Classe du passager
    "Sex",        # Sexe
    "Age",        # Âge
    "SibSp",      # Nombre de frères et sœurs/époux à bord
    "Parch",      # Nombre de parents/enfants à bord
    "Fare",       # Tarif payé
    "Embarked"    # Port d'embarquement
)

# Supprimer les lignes avec des valeurs nulles
df = df.dropna()

# Encodage des variables catégoriques (ex. Sex, Embarked)
sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex")
df = sex_indexer.fit(df).transform(df)
df = embarked_indexer.fit(df).transform(df)

# Assembleur pour créer un vecteur de features
assembler = VectorAssembler(
    inputCols=["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex"],
    outputCol="features"
)
df = assembler.transform(df)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Modèle de régression logistique pour la classification binaire
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_data)

# Évaluation sur l'ensemble de test
predictions = model.transform(test_data)

# Évaluation des performances du modèle
evaluator = BinaryClassificationEvaluator(
    labelCol="label", 
    rawPredictionCol="rawPrediction", 
    metricName="areaUnderROC"
)
roc_auc = evaluator.evaluate(predictions)
print(f"Area Under ROC Curve (AUC): {roc_auc}")

# Résultats du modèle
predictions.select("Pclass", "Sex", "Age", "Fare", "label", "prediction").show(10)

# Arrêter la session Spark
spark.stop()
