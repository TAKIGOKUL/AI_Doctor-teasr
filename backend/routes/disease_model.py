import uvicorn
from fastapi import APIRouter
from twilio.rest import Client
from dotenv import load_dotenv
import os
import numpy as np
import pickle
import pandas as pd
import pywhatkit
disease_api_router = APIRouter()

# models
from models.Blood_Cell.Blood_Cell import Blood_Cell
from models.Breast_Cancer.BREAST_CANCER import Breast_Cancer
from models.Fever.Fever import Fever
from models.Diabetes.Diabetes import Diabetes
from models.Liver.Liver import Liver

load_dotenv()
# account_sid = os.environ['TWILIO_SID']
# auth_token = os.environ['TWILIO_AUTH']
# msgClient = Client(account_sid, auth_token)

# first time global declaration
disease_to_predict = Breast_Cancer

allModels = [

    # {
    #     "name": "Breast_Cancer",
    #     "model": Breast_Cancer,
    #     "classifier": pickle.load(open("models/Breast_Cancer/b_cancer.pkl","rb"))
    # },
    {
        "name": "Fever",
        "model": Fever,
        # "classifier": pickle.load(open("models/Cough/cough.pkl","rb"))
    },
    {
        "name": "Blood_Cell",
        "model": Blood_Cell,
        "classifier": pickle.load(open("models/Blood_Cell/blood_cell_disease.pkl","rb"))
    },
    {
        "name": "Diabetes",
        "model": Diabetes,
        "classifier": pickle.load(open("models/Diabetes/diabetes.pkl","rb"))
    },
    {
        "name": "Liver",
        "model": Liver,
        "classifier": pickle.load(open("models/Liver/liver.pkl","rb"))
    }
]

@disease_api_router.get('/disease/{disease_name}')
def all_disease_fields(disease_name:str):
    for model in allModels:
        if(model['name'] == disease_name):
            fields = [field.name for field in model['model'].__fields__.values()]
            return fields


# defining for cough
@disease_api_router.post('/predict/Fever')
def predict_cough(data:Fever):
    data = data.dict()
    cough=data['cough']
    # fever=data['fever']
    sore_throat=data['sore_throat']
    shortness_of_breath=data['shortness_of_breath']
    head_ache=data['head_ache']
    age_60_and_above=data['age_60_and_above']
    # now write a if else condition considering the above values
    status="error"
    if(cough==1  and sore_throat==1 and shortness_of_breath==1 and head_ache==1 and age_60_and_above==1):
        prediction="Its a fever, i Recommend you to consult a doctor"

    elif(cough==1 and sore_throat==1 and shortness_of_breath==1 and head_ache==1 and age_60_and_above==0):
        prediction="Its a fever, i Recommend you to take a paracetamol"
    elif(cough==1  and sore_throat==1 and shortness_of_breath==1 and head_ache==0 and age_60_and_above==1):
        prediction="Its a fever"
    elif(cough==1  and sore_throat==1 and shortness_of_breath==1 and head_ache==0 and age_60_and_above==0):
        prediction="Its a fever"
    elif(cough==1  and sore_throat==1 and shortness_of_breath==0 and head_ache==1 and age_60_and_above==1):
        prediction="Its a cold or cough"
    elif(cough==1  and sore_throat==0 and shortness_of_breath==0 and head_ache==1 and age_60_and_above==0):
        prediction="Its a  mild cold or cough"
    elif (cough==0  and sore_throat==0 and shortness_of_breath==0 and head_ache==0 and age_60_and_above==1):
        prediction= "No symnptoms of fever"
    else:
        prediction="No symptoms of fever"
        status = "success"
    pywhatkit.sendwhatmsg_instantly(phone_no="+917840073450",message="I hope this message finds you well. I wanted to reach out and remind you of the importance of taking care of your health, especially when it comes to Blood Cell Disease. Early detection and treatment can make a significant impact on your outcome. I strongly recommend that you schedule an appointment with a doctor who specializes in Blood Cell Disease. I have included their contact information below. Doctor's Name: Dr. Prajapati Phone Number: +935646556")
    return {
        'prediction': prediction,
        'status': status
    }
    

    



# @disease_api_router.post('/predict/Breast_Cancer')
# def predict_breast_cancer(data:Breast_Cancer):
#     data = data.dict()
#     clump_thickness=data['clump_thickness']
#     uniform_cell_size=data['uniform_cell_size']
#     uniform_cell_shape=data['uniform_cell_shape']
#     marginal_adhesion=data['marginal_adhesion']
#     single_epithelial_size=data['single_epithelial_size']
#     bare_nuclei=data['bare_nuclei']
#     bland_chromatin=data['bland_chromatin']
#     normal_nucleoli=data['normal_nucleoli']
#     mitoses=data['mitoses']
#    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
#     prediction_val = pickle.load(open("models/Breast_Cancer/b_cancer.pkl","rb")).predict([[clump_thickness,uniform_cell_size,uniform_cell_shape,marginal_adhesion,single_epithelial_size,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses]])
#     if(prediction_val[0]==2):
#         prediction="Its a Breast Cancer"
#         status = "error"
#         # msgClient.messages.create(
#         # body= f"I hope this message finds you well. I wanted to reach out and remind you of the importance of taking care of your health, especially when it comes to breast cancer. Early detection and treatment can make a significant impact on your outcome. I strongly recommend that you schedule an appointment with a doctor who specializes in breast cancer. I have included their contact information below. Doctor's Name: Dr. Prajapati Phone Number: +935646556",
#         # from_="+19298224131",
#         # to="+919301912689"
#         # )   
#     elif(prediction_val[0]==4):
#         prediction="Hurray you got no cancer"
#         status = "success"
#     return {
#         'prediction': prediction,
#         'status': status
#     }


@disease_api_router.post('/predict/Blood_Cell')
def predict_blood_cell_disease(data:Blood_Cell):
    data = data.dict()
    pelvic_incidence=data['pelvic_incidence']
    pelvic_tilt=data['pelvic_tilt']
    lumbar_lordosis_angle=data['lumbar_lordosis_angle']
    sacral_slope=data['sacral_slope']
    pelvic_radius=data['pelvic_radius']
    grade_of_spondyolistesis=data['grade_of_spondyolistesis']

    prediction_val = pickle.load(open("models/Blood_Cell/blood_cell_disease.pkl","rb")).predict([[pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,sacral_slope,pelvic_radius,grade_of_spondyolistesis]])
    if(prediction_val[0]==1):
        prediction="Its a Blood cell DH disease"
        status = "error"
        pywhatkit.sendwhatmsg_instantly(phone_no="+917840073450",message="I hope this message finds you well. I wanted to reach out and remind you of the importance of taking care of your health, especially when it comes to Blood Cell Disease. Early detection and treatment can make a significant impact on your outcome. I strongly recommend that you schedule an appointment with a doctor who specializes in Blood Cell Disease. I have included their contact information below. Doctor's Name: Dr. Prajapati Phone Number: +935646556")
        # msgClient.messages.create(
        # body= f"I hope this message finds you well. I wanted to reach out and remind you of the importance of taking care of your health, especially when it comes to Blood Cell Disease. Early detection and treatment can make a significant impact on your outcome. I strongly recommend that you schedule an appointment with a doctor who specializes in breast cancer. I have included their contact information below. Doctor's Name: Dr. Prajapati Phone Number: +935646556",
        # from_="+19298224131",
        # to="+919301912689"
        # )  
    elif(prediction_val[0]==2):
        prediction="It is a SH"
        status = "warning"

    elif(prediction_val[0]==3):
        prediction="Hurray you got no Blood cell DH disease"
        status = "success"
    return {
        'prediction': prediction,
        'status': status
    }


@disease_api_router.post('/Diabetes')
def predict_diabetes(data:Diabetes):
    data=data.dict()
    Pregnancies=data['Pregnancies']
    Glucose=data['Glucose']
    BloodPressure=data['BloodPressure']
    SkinThickness=data['SkinThickness']
    Insulin=data['Insulin']
    BMI=data['BMI']
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    Age=data['Age']

    predction_val=pickle.load(open("models/Diabetes/diabetes.pkl","rb")).predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if(predction_val[0]==0):
        prediction="Hurray you got no Diabetes"
        status = "success"
    elif(predction_val[0]==1):
        prediction="Its a Diabetes"
        status = "error"
        # msgClient.messages.create(
        # body= f"I hope this message finds you well. I wanted to reach out and remind you of the importance of taking care of your health, especially when it comes to Diabetes. Early detection and treatment can make a significant impact on your outcome. I strongly recommend that you schedule an appointment with a doctor who specializes in breast cancer. I have included their contact information below. Doctor's Name: Dr. Prajapati Phone Number: +935646556",
        # from_="+19298224131",
        # to="+919301912689"
        # )  
    return {
        'prediction': prediction,
        'status': status
    }

@disease_api_router.post('/predict/Liver')
def predict_heart_disease(data:Liver):
    data=data.dict()
    age=data['age']
    gender=data['gender']
    total_bilirubin=data['total_bilirubin']
    direct_bilirubin=data['direct_bilirubin']
    alkaline_phosphotase=data['alkaline_phosphotase']
    alamine_aminotransferase=data['alamine_aminotransferase']
    aspartate_aminotransferase=data['aspartate_aminotransferase']
    total_protiens=data['total_protiens']
    albumin=data['albumin']
    albumin_and_globulin_ratio=data['albumin_and_globulin_ratio']


    predction_val=pickle.load(open("models/Liver/liver.pkl","rb")).predict([[age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_protiens,albumin,albumin_and_globulin_ratio]])
    if(predction_val==0):
        prediction="Its a LIver Disease",
        status = 'error'
        # msgClient.messages.create(
        # body= f"I hope this message finds you well. I wanted to reach out and remind you of the importance of taking care of your health, especially when it comes to Liver Problems. Early detection and treatment can make a significant impact on your outcome. I strongly recommend that you schedule an appointment with a doctor who specializes in breast cancer. I have included their contact information below. Doctor's Name: Dr. Prajapati Phone Number: +935646556",
        # from_="+19298224131",
        # to="+919301912689"
        # )  
    elif(predction_val==1):
        prediction="Hurray you got liver Disease"
        status = 'success'
    return {
        'prediction': prediction,
        'status': status
    }