from flask import Flask,render_template, request
import numpy as np
import pickle
import logging

from src import logger
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



app=Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status',methods=['GET','POST'])
def status():
    msg=""
    if request.method=='POST':
        try:
            footfall=request.form.get("footfall")
            tempMode=request.form.get("tempmode")
            AQ=request.form.get("aq")
            USS=request.form.get("uss")
            CS=request.form.get("cs")
            VOC=request.form.get("voc")
            RP=request.form.get("rpm")
            IP=request.form.get("ip")
            Temperature=request.form.get("tp")

            logger.info("Data recieved: %s, %s, %s, %s, %s, %s, %s, %s, %s",footfall,tempMode,AQ,USS,CS,VOC,RP,IP,Temperature)

            footfall=int(footfall)
            RP=int(RP)
            
            data=CustomData(
                footfall=footfall,
                tempMode=tempMode,
                AQ=AQ,
                USS=USS,
                CS=CS,
                VOC=VOC,
                RP=RP,
                IP=IP,
                Temperature=Temperature
            )

            pred_df=data.get_data()
            logger.info("Dataframe column: %s",pred_df.columns)
            logger.info("Dataframe: %s",pred_df)

            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)[0]

            if results==0:
                msg="Under the given conditions, Machine Failure is not possible"
            else:
                msg="Under the given conditions, Machine Failure is possible" 

            return render_template("status.html",msg=msg)       
            

        except Exception as e:
            logger.error("Error during prediction: %s",e)
            return render_template("index.html")
    else:
        return render_template('status.html',msg=msg)
    

#driver code
if __name__=="__main__":
    app.run(debug=True)