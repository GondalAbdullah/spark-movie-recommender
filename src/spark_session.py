from pyspark.sql import SparkSession

def get_spark(app_name="MovieRecSpark", memory="4g"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "200")
        .getOrCreate()
    )
    return spark
