import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, udf
from pyspark.sql import functions as F

# определяем является ли тип объявления CPM
func_is_cpm = udf(lambda x: 1 if x == 'CPM' else 0)

# определяем является ли тип объявления CPC
func_is_cpc = udf(lambda x: 1 if x == 'CPC' else 0)


# добавляем колонку day_count
def func_day_count_add(input_file):
    # создаем DF - группируем исходный по ad_id, для событий с просмотрами,
    # находим для каждого ad_id timestamp первого и последнего просмотров
    df_timestamps = input_file.where(col('event') == 'view').groupBy('ad_id') \
        .agg(F.min('date').alias('timestamp_first_view'), \
             F.max('date').alias('timestamp_last_view'))

    # добавляем к созданному DF колонку с количеством дней, когда показывалась реклама,
    # считаем разницу в днях между первым и последним показом
    df_day_count = df_timestamps.withColumn('day_count', datediff(col('timestamp_last_view'), \
                                                                  col('timestamp_first_view')))

    # делаем outer join исходного DF и колонки day_count из нового DF по ad_id, в результате для некоторых объявлений
    # в колонке day_count будут Null значения, что является маркером того, что для данного ad_id
    # в БД отражены события связанные с кликами
    df_joined = input_file.join(df_day_count, ['ad_id'], how='outer').orderBy('day_count').fillna({'day_count': 0})

    return df_joined


# добавляем колонку CTR
def func_CTR_add(input_file):
    # создаем DF - группируя исходный DF по ad_id и считаем количество кликов для каждого
    df_clicks = input_file.where(col('event') == 'click').groupBy('ad_id').agg(F.count('event').alias('clicks_number'))

    # создаем DF - группируя исходный DF по ad_id и считаем количество показов для каждого
    df_views = input_file.where(col('event') == 'view').groupBy('ad_id').agg(F.count('event').alias('views_number'))

    # объединяем полученные DF-мы с исходным DF (т.е. добавляем 2 новые колонки - клики и показы), null в соответствующих
    # колонках заполняем 0-ми
    df_joined = input_file.join(df_clicks, ['ad_id'], how='outer') \
        .join(df_views, ['ad_id'], how='outer') \
        .fillna({'clicks_number': 0, 'views_number': 0})

    # добавляем колонку CTR к итоговому DF и переопределяем его с тем же именем
    df_joined = df_joined.withColumn('CTR', col('clicks_number') / col('views_number'))

    return df_joined


# последовательно добавляем все новые колонки, получаем итоговый DF
def transform_pipeline(input_file):
    # добавляем колонку day_count
    new_input_file = func_day_count_add(input_file)

    # добавляем колонку CTR
    new_input_file = func_CTR_add(new_input_file)

    # добавляем колонки is_cpm и is_cpc
    new_input_file = new_input_file.withColumn('is_cpm', func_is_cpm(col('ad_cost_type'))) \
        .withColumn('is_cpc', func_is_cpc(col('ad_cost_type')))

    return new_input_file


# делаем разбиение, записываем результат по соответствующему пути
def get_result(DF, path):
    splits = DF.randomSplit([0.5, 0.25, 0.25], seed=0)

    splits[0].coalesce(1).write.option('header', 'true').parquet(str(path) + '/train')
    splits[1].coalesce(1).write.option('header', 'true').parquet(str(path) + '/test')
    splits[2].coalesce(1).write.option('header', 'true').parquet(str(path) + '/validate')






def process(spark, input_file, target_path):

    input_file = spark.read.parquet('input_file')

    DF = transform_pipeline(input_file)[
        ['ad_id',
         'target_audience_count',
         'has_video',
         'is_cpm',
         'is_cpc',
         'ad_cost',
         'day_count',
         'CTR']
    ]

    get_result(DF.distinct(), 'target_path')



def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
