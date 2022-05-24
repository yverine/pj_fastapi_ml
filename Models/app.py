import uvicorn
from fastapi import FastAPI, Query
import joblib

# init app
app = FastAPI()


# Vectorizer
gender_vectorizer = open("gender_vectorizer.pkl","rb")
gender_cv = joblib.load(gender_vectorizer)

#load models
gender_nv_model = open("gender_nv_model.pkl", "rb")
gender_clf =  joblib.load(gender_nv_model)

#Routes
@app.get("/")
async def index():
    return {"enter": "hello word"}

@app.get("/items/{name}")
async def list(name):
    return {'name': name}


@app.get('/predict/')
async def predict(name:str = Query(None,min_length=2,max_length=12)):
	vectorized_name = gender_cv.transform([name]).toarray()
	prediction = gender_clf.predict(vectorized_name)
	if prediction[0] == 0:
		result = "female"
	else:
		result = "male"

	return {"orig_name":name,"prediction":result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",port=8000)