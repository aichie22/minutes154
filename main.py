from flask import Flask, jsonify, request
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
allCondition = ['Psoriasis', 'Dry_Skin', 'Normal_Skin', 'Itchy_Skin', 'Sensitive_Skin', 'Inflammed_Skin', 'Infected_Skin', 'Dry_Scalp', 'Itchy_Scalp', 'Sensitive_Scalp', 'Oily_Scalp', 'Red_Scalp', 'Flaky_Scalp', 'Wounds', 'Burns', 'Blisters', 'Cuts', 'Red_Skin', 'Delicate_Skin', 'Sun_Rays_Protection', 'Hair', 'Body', 'Face', 'Scalp', 'Mouth', 'Babies', 'Children', 'Adults']
product = ['Pso-Rest Cream', 'Daily Intensive Moisturising Cream', 'Daily Advance Intensive Barrier Repair Cream', 'Gentle Hair Shampoo', 'Calming Body Wash', 'Scalp Repair Spray', 'Daily Resurging Face Serum', 'Hydrating Anti-Photoaging Sunscreen', 'Calming Baby Balm', 'Cooling Snow Cream', 'Moinsturising Cleanser', 'Hydrating Anti-Bacterial Body Wash', 'Facial Cleanser', 'Revitalising Life Essence Mist', 'Lightweight Moisture Booster', 'Aurora Silver Skin Spray', "R'Cares E Squalance Hydrating Oil", 'EZ Anti-Dandruff Shampoo', 'EZ Clean Body Wash']

# with open("trained_model.pkl", "rb") as f:
#   model_product = pickle.load(f)

class PredictionItem(BaseModel):
    condition1: int
    condition2: int
    condition3: int
    condition4: int
    condition5: int
    condition6: int
    condition7: int
    condition8: int
    condition9: int
    condition10: int
    condition11: int
    condition12: int
    condition13: int
    condition14: int
    condition15: int
    condition16: int
    condition17: int
    condition18: int
    condition19: int
    condition20: int
    condition21: int
    condition22: int
    condition23: int
    condition24: int
    condition25: int
    condition26: int
    condition27: int
    condition28: int

with open("trained_model.pkl", "rb") as f:
  model_product = pickle.load(f)

@app.post("/")
def prediction_endpoint(item:PredictionItem):
  df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
  yhat=model_product.predict(df)
  return {"prediction":int(yhat)}


# @app.route("/body", methods=['GET', 'POST'])
# def product_prediction():
#   if request.method == 'POST':
#     data = request.json
#     new_data=[data["condition1"], data["condition2"], data["condition3"], data["condition4"], data["condition5"], data["condition6"], data["condition7"],data["condition8"], data["condition9"], data["condition10"],
#     data["condition11"], data["condition12"], data["condition13"], data["condition14"], data["condition15"], data["condition16"], data["condition17"],data["condition18"], data["condition19"], data["condition20"],
#     data["condition21"], data["condition22"], data["condition23"], data["condition24"], data["condition25"], data["condition26"], data["condition27"]]
#     new_data=pd.DataFrame([new_data], columns = allCondition)
#     res = model_product.predict(new_data)
#     response = {'code':200, 'status':'OK', 
#                 'result':{'prediction': str(res[0])}}  
#     return jsonify(response)




