import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression , DecisionTreeRegressor, RandomForestRegressor,GBTRegressor
from pyspark.sql import SparkSession
import random

#LR_MODEL = 'lr_model'



def estimator_pipeline(train_dataframe, test_dataframe):

    random.seed(0)

    #вектор features
    vector = VectorAssembler(inputCols=train_dataframe.columns[:-1], outputCol='features')

    #estimator LR с параметрами из задания
    estimator_LR = LinearRegression(featuresCol='features', labelCol='ctr',maxIter=40,regParam=0.4,elasticNetParam=0.8)
    #другие эстиматоры с параметрами по умолчанию
    estimator_DT = DecisionTreeRegressor(featuresCol='features',labelCol='ctr')
    estimator_RF = RandomForestRegressor(featuresCol='features',labelCol='ctr')
    estimator_GB = GBTRegressor(featuresCol='features',labelCol='ctr')

    #evaluator
    RMSE_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='ctr', metricName='rmse')

    #спсиок моделей и непосредственно результаты будем записывать в списки
    models_ = []
    RMSE_result = []

    #обучаем все эстиматоры
    for est_r in [estimator_LR, estimator_DT, estimator_RF, estimator_GB]:

        #задаем pipline обучения (2 стадии, в реальности - доп.ступени отчистки и предобработки данных)
        pipeline=Pipeline(stages=[vector,est_r])
        #делаем fit для Pipline по тренировочному датасету (создаем вектор, обучаем эстиматор)
        model = pipeline.fit(train_dataframe)
        #добавляем  модель в список
        models_.append(model)

#       #сохраняем модель (по заданию) - можем сохранть модель в цикле с uid, но тогда нет понимания как правильно
#       #обращатсья к модели через PipelineModel.load из PySparkMLPredict т.к. uid будет постоянно меняться
#       #по этому сохраняем вс модели в список и далее для каждую модель сохраняем с определенным названием
#        model.save(est_r.uid)

        #делаем прогноз по тестовому датасету
        prediction=pipeline.fit(train_dataframe).transform(test_dataframe)
        #считаем метрику RMSE для тестового датасета
        RMSE = round(RMSE_evaluator.evaluate(prediction), 4)
        #записываем результат в массив для отображения в консоли
        RMSE_result.append(RMSE)

    #сохранение моделей
    for pair in zip(models_,['LR_model','DT_model','RF_model','GB_model']):
        pair[0].save(pair[1])

    return models_, RMSE_result


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    #test_data - путь к файлу с данными для оценки качества модели
    #Ваш код

    train_data=spark.read.parquet(train_data)
    test_data=spark.read.parquet(test_data)

    models, results = estimator_pipeline(train_data,test_data)

    #вывод результатов в консоли для каждой модели (добавил отступы, чтобы отделть вывод от остальных логов)
    #после вывода отображаются логи закрытия сессии и отчистки аккумулятора

    print('\n\n\n')
    for pair in zip(['LR_model','DT_model','RF_model','GB_model'], results):
        print('Model: {}, \tRMSE = {}'.format(pair[0], pair[1]))
    print('\n\n\n')


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)

