from flask import Flask, render_template, request, jsonify

app = Flask(__name__)



@app.route('/', methods=['POST','GET'])
def index():
    return render_template("moonkight.html")

if __name__ == '__main__':
    app.run(debug=True)