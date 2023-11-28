from flask import Flask,render_template
from flask_restful import Resource,Api
from flask_cors import CORS
import os
    

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

if __name__=='__main__':
    app.run(debug=True,port=5000)