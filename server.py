from flask import Flask
from flask_restful import Resource,Api
from flask_cors import CORS
import os
import backend.model 

class BasicAPI(Resource):
    def get(self):
        return {'data':backend.model.predict(mode='data')},200
    

app=Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(16).hex()
CORS(app,resources={r"*": {"origins": "*"}})
api=Api(app)

api.add_resource(BasicAPI,'/get_api')




@app.route('/')
def home():
    return '1'


if __name__=='__main__':
    app.run(debug=True,port=5600)