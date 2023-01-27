from flask import Flask, render_template, Response, request, redirect, make_response, jsonify
import cv2
import dlib
from time import strftime, gmtime
import numpy as np
import os
import face_recognition
import math
import shutil
import requests
from PIL import Image
import pathlib
import time

app = Flask(__name__, static_folder='D:\\Programming\\Workbooks\\01. Kazee\\Web\\static')
app.debug = True

camera = cv2.VideoCapture(0)

status = {'faces': "No Face Detected", "confidence": "0", "match-status": False, "error-status": 1}

takePhotoReq = False

detector = dlib.get_frontal_face_detector()

recentPicTaken = ""

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def gen_frames():
    global takePhotoReq,recentPicTaken
    while True:
        success, frame = camera.read()  # read the camera frame
        gray = cv2.cvtColor(np.float32(frame), cv2.COLOR_RGB2GRAY)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            coloredframe = frame
            frame = buffer.tobytes()
            if takePhotoReq:
                timeNow = strftime("%d-%b-%y.%H-%M-%S", gmtime())
                cv2.imwrite(f'static/images/{timeNow}.png',coloredframe)
                recentPicTaken = f'{timeNow}.png'
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []

def encode_faces():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names
    for image in os.listdir('static/faces'):
        face_image = face_recognition.load_image_file(f"static/faces/{image}")
        try:
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(image)
        except IndexError:
            pass

encode_faces()

def recog(frame):
    global status

    status["faces"] = "No Face Detected"
    status["confidence"] = "0%"
    status["match-status"] = False
    status["error-status"] = 1

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Upscale image resolution
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    
    path = 'FSRCNN_x4.pb'
    sr.readModel(path)
    sr.setModel("fsrcnn", 4)
    small_frame = sr.upsample(small_frame)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = '???'

        # Calculate the shortest distance to face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])

        status["faces"] = name
        status["confidence"] = confidence

        face_names.append(f'{name} ({confidence})')

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1

        # Create the frame with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    return frame

@app.route('/')
def index():
    global takePhotoReq, recentPicTaken
    if takePhotoReq:
        takePhotoReq = False
        return redirect("/result?fn="+recentPicTaken)
    return render_template('index.html', status=status)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/getPhoto")
def getPhoto():
    global takePhotoReq
    takePhotoReq = True
    return redirect("/")

@app.route("/result")
def result():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names

    filename = request.args.get('fn')

    filename = os.path.join("static/images/", filename)
    frame = cv2.imread(filename)

    try:
        if filename == None:
            return "<h2>Not a valid file name</h2>"
    except ValueError:
        pass

    frame = recog(frame)

    filename = filename[14:]

    cv2.imwrite(f'static/images/det-{str(filename)}',frame)

    resultFileName = "det-"+str(filename)

    return render_template("result.html", resultFileName=resultFileName, status=status)

@app.route("/api")
def api():
    picLink = request.args.get("l")

    if picLink == None or picLink == "":
        return "<h2>No link argument found</h2>"
    response = requests.get(picLink, stream=True)
    timeNow = strftime("%d-%b-%y.%H-%M-%S", gmtime())
    filename = f"static/images/api-{timeNow}.png"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

    frame = cv2.imread(f"static/images/api-{timeNow}.png")

    try:    
        if filename == None:
            return "<h2>Not a valid file name</h2>"
    except ValueError:
        pass

    global status

    status["faces"] = "No Face Detected"
    status["confidence"] = "0%"
    status["match-status"] = False
    status["error-status"] = 1

    # Upscale image
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'FSRCNN_x4.pb'
    sr.readModel(path)
    sr.setModel("fsrcnn", 4)
    
    if frame.shape[0] >= 1000 or frame.shape[1] >= 1000:
        small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    elif frame.shape[0] <= 400 or frame.shape[1] <= 400 :
        small_frame = sr.upsample(frame)
        small_frame = cv2.resize(small_frame, (0, 0), fx=0.5, fy=0.5)
    else :
        small_frame = frame
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if rgb_small_frame.shape[0] > 600 :
        response = make_response(jsonify({"faceDetected": faceDetected, "confidence": confidence, "match-status": status["match-status"], "error-status": 0}))
        return response

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = '???'

        # Calculate the shortest distance to face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])

        status["faces"] = name
        status["confidence"] = confidence

        if picLink[:58] == "https://waktoo-selfie.obs.ap-southeast-3.myhuaweicloud.com" or picLink[:60] == "https:\/\/waktoo-selfie.obs.ap-southeast-3.myhuaweicloud.com":
            checkedID = picLink.replace("\\", "").split("/")[-1].split("_")[0]
            detectedFace = status["faces"].split(".")[0]
            if checkedID == detectedFace:
                status["match-status"] = True
            else:
                status["match-status"] = False

        face_names.append(f'{name} ({confidence})')

    # Display the results

    faceDetected = status["faces"]
    confidence = status["confidence"]

    response = make_response(jsonify({"faceDetected": faceDetected, "confidence": confidence, "match-status": status["match-status"], "error-status": 1}))

    return response

@app.route("/update")
def update():
    r = requests.get('https://web.waktoo.com/open-api/get-selfie', headers={'Accept': 'application/json'})

    response = r.json()
    idPerusahaan = 1 # PT Kazee Digital Indonesia
    response = response["data"][idPerusahaan-1]["user"]

    for i in response:
        count = 1
        try:
            for j in i["foto"]:
                url = j["foto_absen"]

                r = requests.get(url)

                filename = f'static/faces/{i["user_id"]}.png'

                with open(filename, 'wb') as f:
                    f.write(r.content)         
                try :
                    img = cv2.imread(filename)
                    # Convert into grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Load the cascade
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.4, 6)
                    
                    # Draw rectangle around the faces and crop the faces
                    for (x, y, w, h) in faces:
                        faces = img[y:y + h, x:x + w]
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    path = 'FSRCNN_x4.pb'
                    sr.readModel(path)
                    sr.setModel("fsrcnn", 4)
                    upscaled = sr.upsample(faces)
                    cv2.imwrite(filename, upscaled)
                    break
                except :
                    pass  
                os.remove(filename)         
        except IndexError:
            print("jumlah foto: 0")
    
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)