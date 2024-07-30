import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initialise_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Strating Model Trainer")

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1], #taking all rows and cols except last col
                train_arr[:,-1],  #taking all rows and only last col
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                'SVC':SVC(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'RandomForestClassifier':RandomForestClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                'XGBClassifier':XGBClassifier()
                }
            params={
                "SVC":{
                    'C':[1.0,5.0,10.0,15.0,30.0,50.0],
                    'kernel':['rbf'],
                    'gamma':[0.1,0.2,0.5,0.8,1.0],
                },
                "DecisionTreeClassifier":{
                    'criterion':['gini'],
                    'max_depth':[3,5,10,15,20],
                    'min_samples_split':[2,4,6,8,10],
                    'min_samples_leaf':[1,5,10],
                    'splitter':['best','random'],
                },
                "RandomForestClassifier":{
                    "n_estimators":[50,100,120,150,200],
                    'criterion':['gini'],
                    "max_depth":[2,3,5,8,10,20],
                    'min_samples_split':[2,4,6,7,8,9,10],
                },
                "AdaBoostClassifier":{
                    'n_estimators':[100,200,230,250,],
                    'learning_rate':[0.01,0.05,0.1,0.5,1.0],
                    'algorithm':['SAMME.R'],
                },
                "GradientBoostingClassifier":{
                    'learning_rate':[0.01,0.1,0.2,0.5,0.8,1.0],
                    'n_estimators':[100,200,230,250,300],
                    'max_depth':[3,2,5,8,10],
                    'min_samples_split':[2,3,4,8,10,12],
                },
                "XGBClassifier":{
                    'learning_rate':[0.01,0.1,0.2,0.4,0.6,0.8,1.0],
                    'n_estimators':[120,140,180,200],
                    'max_depth':[1,2,3,5,8,10,15,20],
                    'gamma':[00.1,0.2,0.3,0.5,0.8],
                    'subsample':[0.2,0.4,0.6,0.8,1.0],
                }
            }
            
            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            logging.info(f"Best model found: {best_model}")

            logging.info("Saving the Best Model")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            logging.info("Model saved successfuly")

            predicted=best_model.predict(X_test)
            auc_roc_score=roc_auc_score(y_test,predicted)

            return auc_roc_score

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    try:
        logging.info("Model trainer started")

        from data_ingestion import DataIngestion
        from data_transformation import DataTransformation

        ingestion=DataIngestion()
        train_data_path,test_data_path=ingestion.initiate_data_ingestion()

        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_data_path, test_data_path)

        trainer = ModelTrainer()
        auc_roc_score = trainer.initialise_model_trainer(train_arr, test_arr)

        logging.info(f"Model training completed successfully with auc score: {auc_roc_score}")

    except Exception as e:
        logging.error(f"Error during the model training script execution: {e}", exc_info=True)
        raise        