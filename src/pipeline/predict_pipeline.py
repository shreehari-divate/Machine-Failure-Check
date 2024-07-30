import os
import sys
import joblib
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path=os.path.join('artifacts','model.pkl')
        self.preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file missing")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError("Preprocessor file missing")
        
        self.model=joblib.load(self.model_path)

    def predict(self,features):
        try:
            model=load_object(file_path=self.model_path)
            preprocessor=load_object(file_path=self.preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)    
        
class CustomData:
    def __init__(self,footfall,tempMode,AQ,USS,CS,VOC,RP,IP,Temperature): 
        self.footfall=footfall
        self.tempMode=tempMode
        self.AQ=AQ
        self.USS=USS
        self.CS=CS
        self.VOC=VOC
        self.RP=RP
        self.IP=IP
        self.Temperature=Temperature

    def get_data(self):
        try:
            custom_data_input_dict={
                "footfall":[self.footfall],
                "tempMode":[self.tempMode],
                "AQ":[self.AQ],
                "USS":[self.USS],
                "CS":[self.CS],
                "VOC":[self.VOC],
                "RP":[self.RP],
                "IP":[self.IP],
                "Temperature":[self.Temperature]
            }
            df=pd.DataFrame(custom_data_input_dict)
            print("Custom data frame")
            print(df)
            return df
        except Exception as e:
            raise CustomException(e,sys)