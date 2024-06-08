from flask import Flask, jsonify, render_template, request, redirect, session, url_for
from flask_mail import Mail, Message
import pickle
from models import User
from dal import UserDao
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "@ShopSecret"
mail = Mail(app)  # Initialisation de l'extension Flask-Mail

data = pd.read_csv("C:/Users/ayala/Desktop/Front-Pfa1/Front-Pfa/creditcard.csv")
random_forest_model = RandomForestClassifier()
logistic_regression_model = LogisticRegression()

@app.route("/")
def dashboard():
    try:
        if 'email' in session:
            email = session['email']
            user = UserDao.get_user(email)

            if user:
                if hasattr(user, 'isAdmin') and user.isAdmin:
                    users = UserDao.get_all_users()
                    return render_template('info-carte.html', users=users)

        return render_template('login.html')
    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('templates/login.html')  # Retourne simplement à la page de connexion en cas d'erreur

@app.route("/login", methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    user = UserDao.authenticate_user(email, password)

    if user:
        session['email'] = email
        return redirect(url_for("dashboard"))  # Redirige vers la route 'dashboard' (la fonction dashboard)
    else:
        return render_template('login.html', error='Invalid credentials')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Crée un objet User avec les données fournies
        user = User(email=email, password=password, isAdmin=False)

        # Tente d'enregistrer l'utilisateur
        registration_result = UserDao.register_user(user)

        return redirect('/')

    return render_template('register.html')

@app.route('/info-carte', methods=['GET', 'POST'])
def info_card():
    # Logique de la page 'info-carte'
    return render_template('info-carte.html')

@app.route("/logout")
def logout():
    session.pop('email', None)
    return redirect("/")

###Prediction Fraud - Not Fraud 
@app.route('/predict-realtime', methods=['POST'])
def predict_realtime():
    data = dict(request.get_json(force=True))

    # Chargement du modèle
    with open('Credit.pkl', 'rb') as file:
        pickle_model = pickle.load(file)

    xTest = pd.read_csv('creditcard.csv')

    try:
        time_data = float(data['time'].strip())
        amount_data = float(data['amount'].strip())
    except ValueError as e:
        return jsonify({"result": "Invalid Input"})

    # Obtenir la ligne correspondante
    pca_credit = xTest[(xTest['Time'] == float(data['time'])) & (xTest['Amount'] == float(data['amount']))]

    if len(pca_credit) == 0:
        return jsonify({"result": "Invalid data"})

    required = np.array(pca_credit)
    testData = required[0][:-1].reshape(1, -1)

    output = pickle_model.predict(testData)  # Utilise predict au lieu de decision_function

    if required[0][-1] == 1.0 and output == [-1]:
        result = "It seems to be a Fraudulent Transaction"
    elif required[0][-1] == 0.0 and output == [1]:
        result = "It is a normal Transaction"
    else:
        result = "It seems to be a Fraudulent Transaction"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
