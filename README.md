# Face Recognition Website using Flask and OpenCV.
## Features
1. Live webcam
2. Picture taking
3. Face detection from recent taken picture

## How To Use
Run this project by `flask run` or `python app.py`. After running the program, go to localhost:5000. On the landing page ("/" directory), a list of features will be shown. On "/takepic" route, a webcam will be shown with a "Take Picture" button. When the button is pressed, the picture will be saved in a local folder, and the image will be processed and shown at "/result" directory. Result image will be destroyed on "Back to home" button clicked. On "/live" route, a webcam will also be shown with live face detection. And lastly the "/api" route, GET request will give an upload file frontend and the POST request will process and outputs a JSON.

This commit is NOT docker compatible.

## This is a ready for deployment intern test module for PT Kazee Digital Indonesia, Waktoo Product.
