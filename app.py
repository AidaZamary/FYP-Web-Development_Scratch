from flask import Flask, render_template, request, redirect, url_for, session, flash, Markup
from flask_mysqldb import MySQL
import numpy as np
import pickle
from predictionmodel import DecisionTree
from recommendationmodel import DecisionTreeRecommendation

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'secret'  # Replace with your own secret key

# Database connection details
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'diabetessysdb'
app.config['MYSQL_AUTOCOMMIT'] = True

mysql = MySQL(app)

@app.route('/')
def main():
    return render_template("login.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    cur = mysql.connect.cursor()
    cur.execute("SELECT * FROM user WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()

    if user and user[3] == password:  
        session['user_id'] = user[0] 
        session['loggedin'] = True
        session['name'] = user[1]  
        return redirect(url_for('home'))
    else:
        print("Invalid email or password")
        return render_template('login.html', info='Invalid email or password')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']

        cur = mysql.connect.cursor()

        if not email or not password or not username:
            return render_template('login.html', info='Please fill in all fields')
            
        
        # Check if the user already exists
        cur.execute("SELECT * FROM user WHERE email = %s", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            cur.close()
            return render_template('login.html', info='User already exists. Please choose a different email.')

        # If the user does not exist, proceed with registration
        cur.execute("INSERT INTO user (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
        cur.close()

        # Display success message and redirect to the login page
        return render_template('login.html', info='Registration successful! Please log in.')

    return render_template('login.html')


@app.route('/index')
def home():
    user_name = session.get('name', '')
    if not user_name:
        return redirect(url_for('main'))
    
    # Fetch patient data from database
    cur = mysql.connect.cursor()
    cur.execute("SELECT * FROM patient")
    patient = cur.fetchall()
    cur.close()

    return render_template('index.html', name=user_name, patients=patient)

@app.route('/form', methods=['GET', 'POST'])
def form():
    user_name = session.get('name', '')
    if not user_name or not session.get('loggedin', False):
        return redirect(url_for('main'))

    cur = mysql.connect.cursor()

    if request.method == 'POST':
        patient_id = request.form.get('patient_id')

        if not patient_id:
            return redirect(url_for('prediction'))

        # Fetch patient information for an existing patient
        cur.execute("SELECT * FROM patient WHERE RecordNumber = %s", (patient_id,))
        patient = cur.fetchone()
        cur.close()

        if not patient:
            return redirect(url_for('prediction'))

        return render_template('diagnosis.html', name=user_name, patients=patient)
    
    return render_template('diagnosis.html', name=user_name, patients=None)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    loaded_model = pickle.load(open('decision_tree_prediction.pkl','rb'))
    result = loaded_model.predict(to_predict)
    return result[0]

def ValueRecommender(to_predict_list):
    to_recommend = np.array(to_predict_list).reshape(1,-1)
    load_model = pickle.load(open('decision_tree_recommendation.pkl','rb'))
    recommend = load_model.predict(to_recommend)
    return recommend[0]

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    user_name = session.get('name', '')
    if not user_name or not session.get('loggedin', False):
        return redirect(url_for('main'))
    
    if request.method == 'POST':
       to_predict_dict = request.form.to_dict()
       print(to_predict_dict)
        # Filter out non-integer values and convert the rest to integers
       to_predict_list = [int(value) for key, value in to_predict_dict.items() if value.isdigit()]

       print(to_predict_list)

       result = ValuePredictor(to_predict_list)
       if int(result) == 1:
        prediction = 'Normal Health Condition'
       elif int(result) == 2:
        prediction = 'Prediabetes'
       elif int(result) == 3:
        prediction = 'Diabetes Mellitus (T2DM)'
       elif int(result) == 4:
        prediction = 'Diabetes Mellitus (T2DM) with complications and other disease'
       else:
        prediction = 'Others'

       recommend = ValueRecommender(to_predict_list)
       if int(recommend) == 1:
        recommendation = Markup('<p>Primary Care</p>' +
                                '<p>- Regular physical activity (150 minutes/week)</p>' +
                                '<p>- Prioritize balanced diet</p>' +
                                '<p>- Control sugar intake and unhealthy fats</p>' +
                                '<p>- Regularly monitor blood glucose and health check-ups</p>')
       elif int(recommend) == 2:
        recommendation = Markup('<p>Secondary Care</p>' +
                                '<p>- Physical activity (150 minutes/week)</p>' +
                                '<p>- Prioritize well-balanced diet</p>' +
                                '<p>- Medication:</p>' +
                                '<p>- Oral medication to manage blood glucose level</p>' +
                                '<p>- Insulin therapy (if required)</p>' +
                                '<p>- Patient education on diabetes management and self-care</p>' +
                                '<p>- Regular follow-up appointment</p>' +
                                '<p>- Specific treatment plan</p>')
       else:
        recommendation = 'Others'

    return render_template('recommender.html', name=user_name, prediction=prediction, recommendation=recommendation)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('loggedin', None)
    return redirect(url_for('main'))

if __name__ == '__main__':
    app.debug = True
    app.run()

