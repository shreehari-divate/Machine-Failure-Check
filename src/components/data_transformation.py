import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transform_object(self):
        try:
            logging.info("Data Transformation process started")

            #pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )

            transformers=[
                ('num_pipeline',num_pipeline,['footfall','tempMode','AQ','USS','CS','VOC','RP','IP','Temperature'])
            ]

            preprocessor=ColumnTransformer(transformers)

            logging.info("Preprocesing completed")
            return preprocessor

       
        except Exception as e:
            raise CustomException(e,sys)    
        
    def initiate_data_transformation(self,train_path:str,test_path:str):
        try:
            logging.info("Loading the training and test datasets")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Successfuly read train and test datasets")

            logging.info("Obtaining Preprocessing object")
            preprocess_obj=self.get_data_transform_object()

            target_col='fail'
            
            input_feature_train_df=train_df.drop(columns=[target_col])
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=[target_col])
            target_feature_test_df=test_df[target_col]

            logging.info("Applying preprocessing object on train and test df")
            input_feature_train_arr=preprocess_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocess_obj.transform(input_feature_test_df)

            logging.info("Combining transformed input features with target variable")
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("creating artifacts if not exist")
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocess_obj_file_path),exist_ok=True)

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocess_obj
            )

            logging.info("Preprocessing saved")
            return (
                train_arr,test_arr,self.data_transformation_config.preprocess_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)    
        
if __name__=="__main__":
    try:
        from data_ingestion import DataIngestion
        ingestion=DataIngestion()
        train_path,test_path=ingestion.initiate_data_ingestion()

        transformer=DataTransformation()
        train_arr,test_arr,preprocess_apth=transformer.initiate_data_transformation(train_path,test_path)
        logging.info("Transformation completed successful and preprocessor saved ")        

    except Exception as e:
        raise CustomException(e,sys)    