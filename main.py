from cnnClassifier import logger
from cnnClassifier.pipeline.data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.prepare_model import PrepareBaseModelTrainingPipeline


stage= "Data Ingestion"
try:
        logger.info(f"Stage {stage} started:-------")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f" Stage {stage} completed !")
except Exception as e:
        logger.exception(e)
        raise e


stage = 'Prepare base Model'
try:
        logger.info(f"------------------")
        logger.info(f"Stage: {stage} started:-----")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f"Stage: {stage} completed !")
except Exception as e:
        logger.exception(e)
        raise e