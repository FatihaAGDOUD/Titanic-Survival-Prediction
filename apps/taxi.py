from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialisation de Spark
spark = SparkSession.builder.appName("NYC_Taxi_Trip_Analysis")  \
    .config("spark.master","spark://spark-master:7077")\
    .getOrCreate()

# Charger les données depuis HDFS (ou une autre source)
data_path = "hdfs://namenode:9000/data/tripdata.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Afficher les colonnes pour comprendre la structure des données
df.printSchema()

# Filtrer et sélectionner les colonnes utiles
df = df.select(
    "trip_distance",   # Distance du trajet
    "passenger_count", # Nombre de passagers
    "fare_amount",     # Montant payé
    "payment_type",    # Type de paiement (catégorique)
    "tip_amount",      # Pourboire (valeur cible)
)

# Supprimer les lignes avec des valeurs nulles
df = df.dropna()

# Encodage des variables catégoriques (ex. payment_type)
indexer = StringIndexer(inputCol="payment_type", outputCol="payment_type_index")
df = indexer.fit(df).transform(df)

# Assembleur pour créer un vecteur de features
assembler = VectorAssembler(
    inputCols=["trip_distance", "passenger_count", "fare_amount", "payment_type_index"],
    outputCol="features"
)
df = assembler.transform(df)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Modèle de régression linéaire pour prédire le montant des pourboires
lr = LinearRegression(featuresCol="features", labelCol="tip_amount")
model = lr.fit(train_data)

# Évaluation sur l'ensemble de test
predictions = model.transform(test_data)

# Évaluation des performances du modèle
evaluator = RegressionEvaluator(
    labelCol="tip_amount", 
    predictionCol="prediction", 
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Résultats du modèle
predictions.select("trip_distance", "fare_amount", "tip_amount", "prediction").show(10)

# Arrêter la session Spark
spark.stop()
