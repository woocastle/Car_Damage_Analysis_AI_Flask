from flask import Flask, render_template, request
from app import source

app = Flask(__name__)

# GET /
@app.get('/')
def main():
  return render_template('main.html')

# POST /upload
@app.post('/upload')
def imageUpload():
  file = request.files['image']
  return source.getPrice(file)
