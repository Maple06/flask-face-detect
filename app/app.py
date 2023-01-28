from flask import Flask, render_template, Response, request, redirect, make_response, jsonify
import cv2
from time import strftime, gmtime
import numpy as np
import os

app = Flask(__name__)
app.debug = True

camera = cv2.VideoCapture(0)

takePhotoReq = False

recentPicTaken = ""

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
                cv2.imwrite(f'static/Images/{timeNow}.png',coloredframe)
                recentPicTaken = f'{timeNow}.png'
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global takePhotoReq, recentPicTaken
    if takePhotoReq:
        takePhotoReq = False
        return redirect("/result?fn="+recentPicTaken)
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/getPhoto")
def getPhoto():
    global takePhotoReq
    takePhotoReq = True
    return redirect("/")

@app.route("/result", methods=["GET", "POST"])
def result():
    global recentPicTaken
    if request.method == "POST":
        os.remove(os.path.join("static/Images/", recentPicTaken))
        os.remove(os.path.join("static/Images/", f"det-{recentPicTaken}"))
        return redirect("/")
    else:
        filename = request.args.get('fn')

        filename = os.path.join("static/Images/", filename)

        try:
            if filename == None:
                return "<h2>Not a valid file name</h2>"
        except ValueError:
            pass

        try:
            if filename == None:
                return "<h3>Not a valid filename.<h3><h5><a href='/'>Go back to home</a></h5>"
        except ValueError:
            pass

        # Read recently grabbed image to cv2
        img = cv2.imread(filename)

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale3(gray, 1.1, 5, outputRejectLevels = True)
        try :
            face_detected = list(faces[0].tolist())
            weights = list(faces[2].tolist())
            updated_weights = weights.copy()
        
            # Deleting faces that under 50% confidence
            for i in range(len(weights)) :
                if weights[i] < 5 :
                    face_detected.pop(i)
                    updated_weights.pop(i)
            weights_json = {}
            count = 1
            for i in updated_weights :
                i = '{:.2f}'.format(i*10)
                if float(i) > 100:
                    i = '100.00'
                weights_json[count] = i+"%"
                count += 1

            filename = filename[14:]
            recentPicTaken = filename

            # Drawing a rectangle around detected face(s)
            count = 0
            for (x, y, w, h) in face_detected:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                count += 1
                cv2.putText(img, str(count) , (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

            cv2.imwrite(f'static/Images/det-{str(filename)}',img)
            resultFileName = "det-"+str(filename)
            return render_template("result.html", resultFileName=resultFileName, status={"face-count": len(face_detected), "confidence": weights_json})
        except :
            filename = filename[14:]
            recentPicTaken = filename
            cv2.imwrite(f'static/Images/det-{str(filename)}',img)
            resultFileName = "det-"+str(filename)
            return render_template("result.html", resultFileName=resultFileName, status={"face-count": 0, "confidence": "0%"})

if __name__ == "__main__":
    app.run(debug=True)