import os
import sys
import dill
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)        

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for model_name, model in models.items():
            param_grid=params.get(model_name,{})

            if param_grid:
                gs=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,n_jobs=-1)
                gs.fit(X_train,y_train)
                best_params=gs.best_params_

                model.set_params(**best_params)
            else:
                logging.info(f"No parameters to tune for {model_name}")        
            
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score=roc_auc_score(y_train,y_train_pred)
            test_model_score=roc_auc_score(y_test,y_test_pred)

            report[model_name]=test_model_score

        logging.info("Model evaluation completed")
        return report
    
    except Exception as e:
        raise CustomException(e,sys)    
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
        