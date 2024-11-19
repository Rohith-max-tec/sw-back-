from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import joblib
import pandas as pd
import os
import math
from pymongo import MongoClient
import logging
from datetime import timedelta
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import gc
import tensorflow as tf

from io import BytesIO
import base64


import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pytz



# Initialize the app and setup CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Secret key for session management
app.secret_key = os.urandom(24)

# Set session to be permanent and set a longer session lifetime
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and data
cls = joblib.load('police_up.pkl')
en = joblib.load('label_encoder_up.pkl')
df1 = pd.read_csv('Sih_police_station_data.csv')
model = joblib.load('human_vs_animal.pkl')
df2 = pd.read_csv('districtwise-crime-against-women (1).csv')
df2 = df2[['registeration_circles', 'total_crime_against_women']]

# Define function to classify crime alert
def crime_indicator(crime_count):
    if crime_count < 50:
        return '🟢Green'
    elif 50 <= crime_count <= 500:
        return '🟡Yellow'
    else:
        return '🔴Red'

df2['indicator'] = df2['total_crime_against_women'].apply(crime_indicator)

# MongoDB connection
client = MongoClient("mongodb+srv://rh0665971:q7DFaWad4RKQRiWg@cluster0.gusg4.mongodb.net/?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client['swaraksha']
users_collection = db['users']
otp_collection = db['otp_storage']  
messages_collection = db['messages']
reviews = db['reviews']
App_reviews = db['App_reviews']

# Setup the uploads folder
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize the TensorFlow Lite interpreter for gender prediction
def initialize_interpreter():
    interpreter = tf.lite.Interpreter(model_path="my_gender_final2.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Modify the gender prediction code using TensorFlow Lite
def predict_gender(interpreter, resized_face):
    # Preprocess the face image
    test_img = cv2.resize(resized_face, (64, 64))
    test_img = np.expand_dims(test_img, axis=0).astype(np.float32)

    # Set the input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], test_img)

    # Run the inference
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    # Load the image using OpenCV
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Failed to load image."}), 400

    # Convert image to base64 for MongoDB storage
    with open(filepath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    img_data = f"data:image/jpeg;base64,{encoded_image}"

    

    # Resize image if dimensions exceed 1024x1024
    if img.shape[0] > 1024 or img.shape[1] > 1024:
        img = cv2.resize(img, (1024, 1024))

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        count_male = 0
        count_female = 0

        # Initialize TensorFlow Lite interpreter for gender prediction
        interpreter = initialize_interpreter()

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            # Predict gender using TensorFlow Lite
            y_hat = predict_gender(interpreter, cropped_face)

            if y_hat[0][0] > 0.5:
                count_male += 1
            else:
                count_female += 1

        total_faces = count_male + count_female
        logger.info(f'Number of males: {count_male}, Number of females: {count_female}, Total faces: {total_faces}')


        # Insert the base64-encoded image into MongoDB
        new_message = {
            "message": img_data,
            "male": count_male,
            "female": count_female,
            "total": total_faces,
            "type": "image"
            
        }
        messages_collection.insert_one(new_message)
        logger.info(f"Image inserted into MongoDB as base64 data.")

        # Clean up interpreter and memory after each request
        interpreter = None
        gc.collect()

        return jsonify({
            'num_males': count_male,
            'num_females': count_female,
            'total_faces': total_faces
        })

    return jsonify({"message": "No faces detected in the image."}), 200




@app.route('/community')
def community():
    username = session.get('username', 'Guest')  # Get the username from the session
    logger.info(f"Community page accessed by {username}.")
    return jsonify({"username": username})

@app.route('/getMessages', methods=['GET'])
def get_messages():
    messages = list(messages_collection.find({}, {'_id': 0}))
    messages_list = []
    for msg in messages:
        message_data = {
            "username": msg.get('username', 'Image'),
            "male" : msg.get('male','not found'),
            "female" : msg.get('female','not found'),
            "total" : msg.get('total','not found'),
            "type": msg.get('type', 'text'),
            
        }
        if msg.get('type') == 'audio':
            message_data["filename"] = msg.get('filename')
        else:
            message_data["message"] = msg.get('message', '')
        messages_list.append(message_data)
    return jsonify({"messages": messages_list})

# Route to send a text message
@app.route('/sendMessage', methods=['POST'])
def send_message():
    data = request.json
    username = data.get('username', 'Guest')  # Default to 'Guest' if not provided
    new_message = {
        "message": data['message'],
        "username": username,
        "type": "text"
    }
    messages_collection.insert_one(new_message)
    logger.info(f"Text message sent by {username}: {data['message']}")
    return jsonify({"status": "Message sent!"})

# Route to send a voice message
@app.route('/sendVoiceMessage', methods=['POST'])
def send_voice_message():
    username = request.form.get('username', 'Guest')
    if 'voiceMessage' not in request.files:
        return jsonify({"error": "No voice message provided"}), 400

    voice_file = request.files['voiceMessage']
    filename = secure_filename(voice_file.filename)
    filepath = os.path.join('uploads', filename)
    voice_file.save(filepath)

    # Store the file reference in MongoDB
    new_message = {
        "username": username,
        "type": "audio",
        "filename": filename
    }
    messages_collection.insert_one(new_message)
    logger.info(f"Voice message sent by {username}: {filename}")
    return jsonify({"status": "Voice message sent!"})

# Route to serve uploaded audio files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Route to get the username from the session
@app.route('/getUsername', methods=['GET'])
def get_username():
    username = session.get('username', 'Guest')
    return jsonify({"username": username})

# Route to handle SOS messages (triggered when SOS button is pressed)
@app.route('/sendSOS', methods=['POST'])
def send_sos():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']
    address = data['address']
    username = data['username']
    mobile = data['mobile']
    timezone = pytz.timezone('Asia/Kolkata')
    current_timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
    sos_message = f"Emergency! Please help me at (address: {address}, Latitude: {latitude}, Longitude: {longitude}, mobile: {mobile} Timestamp: {current_timestamp}) or Track me in map https://www.google.com/maps?q={latitude},{longitude}"
    new_message = {
        "message": sos_message,
        "username": username,
        "type": "text"
    }
    messages_collection.insert_one(new_message)
    logger.info(f"SOS message sent by {username}: {sos_message}")
    return jsonify({"status": "SOS sent!"})

  

# Home page route
@app.route('/index')
def index():
    username = session.get('username', 'Guest')
    return render_template('index.html', username=username)

@app.route('/sendSOS2', methods=['POST'])
def send_sos2():
    # Check if image is included in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get image and username from the request
    image = request.files['image']
    username = request.form.get('username', 'Guest')
    
    # Set a secure path for saving the file on disk
    filename = secure_filename(image.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(save_path)  # Save image to the specified path

    # Optionally, convert image to binary for MongoDB storage
    image_binary = BytesIO()
    image.save(image_binary, format=image.format)
    image_data = base64.b64encode(image_binary.getvalue()).decode('utf-8')
    
    # Create the message entry for MongoDB
    new_message = {
        "message": image_data,
        "username": username,
        "type": "image"
    }
    
    messages_collection.insert_one(new_message)  # Store in MongoDB
    logger.info(f"SOS image message sent by {username} and saved at {save_path}")
    
    return jsonify({"status": "SOS sent with image!"})

@app.route('/get_otp', methods=['GET', 'POST'])
def get_otp():
    if request.method == 'POST':
        username = request.form.get('username')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        password = request.form.get('password')


        if not username or not email or not password:
            flash('All fields are required!')
            return redirect(url_for('register'))

        # Check if email already exists
        if users_collection.find_one({'email': email}):
            return jsonify({'success': False, 'message': 'Email already exists! Please log in.'})

        otp_collection.delete_one({"email": email})

        # Generate and store OTP
        otp=send_otp(email)
        if otp is None:
            return jsonify({'success': False, 'message': 'Failed to send OTP. Please try again.'})
        else:
            return jsonify({'success': True, 'message': 'OTP sent to your Email.Please check'})

def send_otp(email):
    otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
    message = f"Your OTP for Swaraksha registration is: {otp}"

    # Email setup
    #sender_email = "rh0665971@gmail.com"
    #sender_password = "jgjr aysx ndto bxye" # Ensure this is a Google App password

    sender_email = "swaraksha247@gmail.com"
    sender_password = "mvvh najj rbee tsfo"


    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = "OTP for Swaraksha Registration"
    msg.attach(MIMEText(message, 'plain'))

    try:
        # Send email using SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())

        # Store OTP in MongoDB with an expiration time of 5 minutes
        otp_data = {
            "email": email,
            "otp": otp,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=5)
        }
        otp_collection.insert_one(otp_data)
        logging.info("OTP sent and saved successfully.")
        return otp  # Return the OTP if successfully sent

    except smtplib.SMTPException as e:
        logging.error(f"Failed to send OTP: {e}")
        return None  # Return None to indicate failure


# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        mobile = request.form.get('mobile')
        guardianNum=request.form.get('guardianNum')
        email = request.form.get('email')
        password = request.form.get('password')
        otp = request.form.get('otp')

        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required!'})

        # Check if email already exists
        if users_collection.find_one({'email': email}):
            return jsonify({'success': False, 'message': 'Email already exists! Please log in.'})

        # Retrieve OTP data for the email
        otp_data = otp_collection.find_one({"email": email})

        # Check if otp_data is None (OTP not found for the email)
        if otp_data is None:
            return jsonify({'success': False, 'message': 'No OTP found for this email. Please request a new OTP.'})

        # Ensure the OTP is correct
        if otp_data["otp"] != int(otp):
            return jsonify({'success': False, 'message': 'Invalid OTP.'})

        # Delete the OTP document from the collection
        otp_collection.delete_one({"email": email})

        # Insert new user into MongoDB
        users_collection.insert_one({
            'username': username,
            'mobile': mobile,
            'guardianNum' : guardianNum,
            'email': email,
            'password': password  # Plain text for now as requested
        })

        return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})

    return render_template('registration.html')

@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    if request.method == 'POST':
        email = request.form.get('email')
    if users_collection.find_one({'email': email}):
        otp_collection.delete_one({"email": email})
        otp=fsend_otp(email)
        if otp is None:
            return jsonify({'success': False, 'message': 'Failed to send OTP. Please try again.'})
        else:
            return jsonify({'success': True, 'message': 'OTP sent to your Email.Please check'})
    return jsonify({'success': False, 'message': 'user not registred.Please register'})

def fsend_otp(email):
    otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
    message = f"Your OTP for Swaraksha registration is: {otp}"

    # Email setup
    sender_email = "swaraksha247@gmail.com"
    sender_password = "mvvh najj rbee tsfo"
    # Ensure this is a Google App password

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = "OTP for password update in Swaraksha"
    msg.attach(MIMEText(message, 'plain'))

    try:
        # Send email using SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())

        # Store OTP in MongoDB with an expiration time of 5 minutes
        otp_data = {
            "email": email,
            "otp": otp,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=5)
        }
        otp_collection.insert_one(otp_data)
        logging.info("OTP sent and saved successfully.")
        return otp  # Return the OTP if successfully sent

    except smtplib.SMTPException as e:
        logging.error(f"Failed to send OTP: {e}")
        return None  # Return None to indicate failure


@app.route('/validate', methods=['GET', 'POST'])
def validate():
    if request.method == 'POST':
        email = request.form.get('email')
        Newpassword = request.form.get('password')
        otp = request.form.get('otp')

        if not email or not Newpassword or not otp:
            return jsonify({'success': False, 'message': 'All fields are required!'})

        # Check if email already exists
        #if users_collection.find_one({'email': email}):
            #return jsonify({'success': False, 'message': 'Email already exists! Please log in.'})

        # Retrieve OTP data for the email
        otp_data = otp_collection.find_one({"email": email})

        # Check if otp_data is None (OTP not found for the email)
        if otp_data is None:
            return jsonify({'success': False, 'message': 'No OTP found for this email. Please request a new OTP.'})

        # Ensure the OTP is correct
        if otp_data["otp"] != int(otp):
            return jsonify({'success': False, 'message': 'Invalid OTP.'})

        # Delete the OTP document from the collection
        otp_collection.delete_one({"email": email})

        users_collection.update_one(
            { 'email': email },
            { '$set': { 'password': Newpassword } }
        )

        # Insert new user into MongoDB
        

        return jsonify({'success': True, 'message': 'New password updated successful! Please log in.'})

    return render_template('forgetpassword.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if user exists
        user = users_collection.find_one({'email': email})

        if user and password == user['password']:
            session['username'] = user['username']
            session['mobile'] = user['mobile']
            session['guardianNum'] = user['guardianNum']
            
            session.permanent = True  # Set session as permanent
            logger.info(f"User {user['username']} logged in successfully.")
          
            return jsonify({'success': True, 'username': user['username'], 'mobile': user['mobile'],'guardianNum': user['guardianNum']})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials!'})

    return render_template('login.html')
@app.route('/add_guardianNum', methods=['POST'])
def add_guardianNum():
    try:
        if request.method == 'POST':
            email = request.form['email']
            guardianNum = request.form['guardianNum']
            logger.info(f'your email:{email} your guardianNum:{guardianNum}')
            # Ensure email and guardianNum are provided
            if not email or not guardianNum:
                logger.info(f'no email{email} or guardianNum found {guardianNum}')
                return jsonify({'success': False, 'message': 'Email and Guardian Number are required'}), 400
            
            # Update the user's guardian number
            result = users_collection.update_one(
                {'email': email},
                {'$set': {'guardianNum': guardianNum}}
            )

            if result.matched_count == 0:
                return jsonify({'success': False, 'message': 'User not found'}), 404

            # Retrieve the updated user document
            user = users_collection.find_one({'email': email})
            guardianNum = user.get('guardianNum')

            return jsonify({'success': True, 'message': 'Saved Successfully', 'guardianNum': guardianNum})

    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'success': False, 'message': 'An error occurred, please try again later'}), 500

@app.route('/get_guardianNum', methods=['POST'])
def get_guardianNum():
    email = request.form.get('email')  # Retrieve email from POST request body
    user = users_collection.find_one({'email': email})
    
    if user and 'guardianNum' in user:
        guardianNum = user.get('guardianNum')
    else:
        guardianNum = None
    
    return jsonify({'guardianNum': guardianNum})
        
    

# Logout route to clear the session
@app.route('/logout', methods=['POST', 'GET'])
def logout():
    username = session.get('username', 'Guest')
    session.clear()  # Clear the session
    logger.info(f"User {username} logged out.")
    return redirect(url_for('index'))

@app.route('/nearestPoliceStation', methods=['POST'])
def nearest_police_station():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    
    # Get guardian number from session, defaulting to 'NF' if not set
    guardianNum = session.get('guardianNum', 'NF')
    
    try:
        # Predict the nearest police station using the trained model
        nearest_police_station.nearest_station = en.inverse_transform(cls.predict([[latitude, longitude]]))
        contact_number = df1.loc[df1['Police_station_name'].str.contains(nearest_police_station.nearest_station[0], case=False, na=False), 'phone_number'].values[0]
        
        # Clean the contact number
        n = contact_number.replace('-', '') 
        logger.info(f"police number is {contact_number} and type is {type(contact_number)}")
        logger.info(f"Guardian number is {guardianNum} and type is {type(guardianNum)}")

        np=nearest_police_station.nearest_station[0]

        station_data = df1[df1['Police_station_name'] == np]

        latitude = station_data['latitude'].values[0]
        longitude = station_data['longitude'].values[0]
        # Return the nearest police station details as JSON
        return jsonify({
            'police_station': nearest_police_station.nearest_station[0],
            'contact_number': n,   # Ensure cleaned contact number is returned
            'guardianNum': guardianNum,
            'latitude': latitude,
            'longitude':longitude
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/distanceP', methods=['POST'])
def distance_p():
    data = request.get_json()
    lat1 = data.get('latitude')
    lon1 = data.get('longitude')
    nearest_station = en.inverse_transform(cls.predict([[lat1, lon1]]))[0]
    lat1 = float(lat1)
    lon1 = float(lon1)

    # Get the nearest station name and location
    station_data = df1[df1['Police_station_name'].str.contains(nearest_station, case=False, na=False)]

    lat2 = station_data['latitude'].values[0]
    lon2 = station_data['longitude'].values[0]

    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    R = 6371000  # Earth's radius in meters
    distance = (R * c) / 1000
    distance = round(distance, 2)

    return jsonify({'police_distance': distance})

@app.route('/emergency', methods=['POST'])
def emergency():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    address = data.get('address')

    # Log received location and address
    logger.info(f'Received emergency location: Latitude {latitude}, Longitude {longitude}, Address {address}')

    return jsonify({'status': 'success', 'latitude': latitude, 'longitude': longitude, 'address': address})

@app.route('/getCrimeAlert', methods=['GET'])
def get_crime_alert():
    city = request.args.get('city')
    crime_alert = 'low'  # Default value
    for i in range(len(df2)):
        if city.lower() in df2['registeration_circles'][i].lower():
            crime_alert = df2['indicator'][i]
            break
    return jsonify({'alert': crime_alert})
    



# Route to accept review data from the frontend
@app.route('/submit_review', methods=['POST'])
def submit_review():
    # Get the JSON data from the request
    data = request.json

    # Extract data from the JSON payload
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    username = data.get('username')
    review_stars = data.get('review_stars')
    review_text = data.get('review_text')

    # Validate input data
    if latitude is None or longitude is None or review_stars is None or review_text is None:
        return jsonify({"error": "Missing data fields"}), 400

    # Create the new message dictionary
    new_message = {
        "username": username,
        "latitude": latitude,
        "longitude": longitude,
        "review_stars": review_stars,
        "review_text": review_text,
        "type": "text"
    }
    
    # Insert the review into the MongoDB collection
    reviews.insert_one(new_message)

    return jsonify({"message": "Review submitted successfully!"}), 201

@app.route('/get_reviews', methods=['POST', 'GET'])
def get_reviews():
    l=[]
    doc=reviews.find()
    for i in doc:
        i['_id'] = str(i['_id'])
        l.append(i)
    return jsonify(l)
    


# Route to accept review data from the frontend
@app.route('/App_review', methods=['POST'])
def App_review():
    # Get the JSON data from the request
    data = request.json

    # Extract data from the JSON payload
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    username = data.get('username')
    review_stars = data.get('review_stars')
    review_text = data.get('review_text')

    # Validate input data
    if latitude is None or longitude is None or review_stars is None or review_text is None:
        return jsonify({"error": "Missing data fields"}), 400

    # Create the new message dictionary
    new_message = {
        "username": username,
        "latitude": latitude,
        "longitude": longitude,
        "review_stars": review_stars,
        "review_text": review_text,
        "type": "text"
    }
    
    # Insert the review into the MongoDB collection
    App_reviews.insert_one(new_message)

    return jsonify({"message": "Review submitted successfully!"}), 201

# Additional emergency and utility routes (trimmed for brevity)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
