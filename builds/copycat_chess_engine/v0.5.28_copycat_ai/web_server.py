from flask import Flask, render_template
import chess
import torch

app = Flask(__name__)

@app.route('/')
def game_interface():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
