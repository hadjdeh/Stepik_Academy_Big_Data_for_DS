import io
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


#LR_MODEL = 'lr_model'


def process(spark, input_file, output_file):
    #input_file - путь к файлу с данными для которых нужно предсказать ctr
    #output_file - путь по которому нужно сохранить файл с результатами [ads_id, prediction]
    #Ваш код

    input_file=spark.read.parquet(input_file)

    for model_name in ['LR_model','DT_model','RF_model','GB_model']:
        model=PipelineModel.load(model_name)
        prediction=model.transform(input_file)
        prediction[['ad_id', 'prediction']].coalesce(1).write.option('header', 'true').csv(output_file+'/'+model_name)


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
