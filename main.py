from cnnClassifier import logger
from cnnClassifier.pipeline.data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.prepare_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.model_trainer import ModelTrainingPipeline


stage= "Data Ingestion"
try:
        logger.info(f"Stage {stage} started:-------")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f" Stage {stage} completed !\n\n")
except Exception as e:
        logger.exception(e)
        raise e


stage = 'Prepare base Model'
try:
        logger.info(f"------------------")
        logger.info(f"Stage: {stage} started:-----")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f" Stage {stage} completed !\n\n")
except Exception as e:
        logger.exception(e)
        raise e


stage = "Training the model"
try:
        logger.info(f"--------------------------")
        logger.info(f"tage {stage} started----------->")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f" Stage {stage} completed !\n\n")
except Exception as e:
        logger.exception(e)
        raise e