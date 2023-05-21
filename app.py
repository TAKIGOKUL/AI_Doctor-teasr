

from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Breast_Cancer(BaseModel):
    clump_thickness: float
    uniform_cell_size: float
    uniform_cell_shape: float
    marginal_adhesion: float
    single_epithelial_size: float
    bare_nuclei: float
    bland_chromatin: float
    normal_nucleoli: float
    mitoses: float

# 1. Library imports
import uvicorn
from fastapi import FastAPI
# from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
# pickle_in = open("/home/vandit/Desktop/hack_this_fall/FastAPI/classifier.pkl","rb")
pickle_in_b_cancer = open("/home/vandit/Desktop/hack_this_fall/b_cancer_api_ready/b_cancer.pkl","rb")
classifier_b_cancer=pickle.load(pickle_in_b_cancer)



# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
""" 
'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses'

"""

@app.post('/breast_cancer')
def predict_breast_cancer(data:Breast_Cancer):
    data = data.dict()
    clump_thickness=data['clump_thickness']
    uniform_cell_size=data['uniform_cell_size']
    uniform_cell_shape=data['uniform_cell_shape']
    marginal_adhesion=data['marginal_adhesion']
    single_epithelial_size=data['single_epithelial_size']
    bare_nuclei=data['bare_nuclei']
    bland_chromatin=data['bland_chromatin']
    normal_nucleoli=data['normal_nucleoli']
    mitoses=data['mitoses']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction_val = classifier_b_cancer.predict([[clump_thickness,uniform_cell_size,uniform_cell_shape,marginal_adhesion,single_epithelial_size,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses]])
    if(prediction_val[0]==2):
        prediction="Its a Breast Cancer"
    elif(prediction_val[0]==4):
        prediction="Hurray you got no cancer"
    return {
        'prediction': prediction
    }

# api for diabetes
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
from pydantic import BaseModel
class diabetes(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float
    # Outcome: float


pickle_in_diabetes = open("/home/vandit/Desktop/hack_this_fall/diabetes_api_working/diabetes.pkl","rb")
classifier_diabetes=pickle.load(pickle_in_diabetes)

@app.post('/diabetes')
def predict_diabetes(data:diabetes):
    data=data.dict()
    Pregnancies=data['Pregnancies']
    Glucose=data['Glucose']
    BloodPressure=data['BloodPressure']
    SkinThickness=data['SkinThickness']
    Insulin=data['Insulin']
    BMI=data['BMI']
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    Age=data['Age']
    # Outcome=data['Outcome']

    predction_val=classifier_diabetes.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if(predction_val[0]==0):
        prediction="Its a No Diabetes"
    elif(predction_val[0]==1):
        prediction="Its a Diabetes"
    return {
        'prediction': prediction
    }
from pydantic import BaseModel
class blood_cell_disease(BaseModel):
    # col=['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle',
    #    'sacral slope', 'pelvic  radius', 'grade of spondyolistesis',
    #    'diagnose']
    pelvic_incidence: float
    pelvic_tilt: float
    lumbar_lordosis_angle: float
    sacral_slope: float
    pelvic_radius: float
    grade_of_spondyolistesis: float
    diagnose: float


# pickle_in = open("/home/vandit/Desktop/hack_this_fall/FastAPI/classifier.pkl","rb")
pickle_in_blood_cell = open("/home/vandit/Desktop/hack_this_fall/blood_cell_disease_api_ready/blood_cell_disease.pkl","rb")
classifier_blood_cell=pickle.load(pickle_in_blood_cell)

@app.post('/blood_cell_disease')
def predict_blood_cell_disease(data:blood_cell_disease):
    data = data.dict()
    pelvic_incidence=data['pelvic_incidence']
    pelvic_tilt=data['pelvic_tilt']
    lumbar_lordosis_angle=data['lumbar_lordosis_angle']
    sacral_slope=data['sacral_slope']
    pelvic_radius=data['pelvic_radius']
    grade_of_spondyolistesis=data['grade_of_spondyolistesis']
    # diagnose=data['diagnose']
    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction_val = classifier_blood_cell.predict([[pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,sacral_slope,pelvic_radius,grade_of_spondyolistesis]])
    if(prediction_val[0]==1):
        prediction="Its a Blood cell DH disease"
    elif(prediction_val[0]==2):
        prediction="It is a SH"
    elif(prediction_val[0]==3):
        prediction="It is a NO"
    return {
        'prediction': prediction 
    }


# Age	Gender	Total_Bilirubin	Direct_Bilirubin	Alkaline_Phosphotase	Alamine_Aminotransferase	Aspartate_Aminotransferase	Total_Protiens	Albumin	Albumin_and_Globulin_Ratio	Dataset
from pydantic import BaseModel
class liver_disease(BaseModel):
    age: float
    gender: float
    total_bilirubin: float
    direct_bilirubin: float
    alkaline_phosphotase: float
    alamine_aminotransferase: float
    aspartate_aminotransferase: float
    total_protiens: float
    albumin: float
    albumin_and_globulin_ratio: float

   

# pickle_in = open("/home/vandit/Desktop/hack_this_fall/FastAPI/classifier.pkl","rb")
pickle_in_liver_disease = open("/home/vandit/Desktop/hack_this_fall/Multi_Disease_Predictor/models/liver.pkl","rb")
classifier=pickle.load(pickle_in_liver_disease)


@app.post('/liver_disease')
def predict_heart_disease(data:liver_disease):
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


    predction_val=classifier.predict([[age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_protiens,albumin,albumin_and_globulin_ratio]])
    if(predction_val==0):
        prediction="Its a LIver Disease"
    elif(predction_val==1):
        prediction="No liver Disease"
    return {
        'prediction': prediction
    }

