# Face Recognition Website using Flask and OpenCV.
## Features
1. Live webcam
2. Picture taking
3. Face detection from recent taken picture

## How To Use
### !! THIS CURRENT COMMIT HAS NOT BEEN TESTED FOR DOCKER YET !!
Build a docker image by `docker build -t <image-name> .` and then make a container with `docker run --name <container-name> -p 8000:8000` and then go to localhost:8000 after running the container, on landing page ("/" directory) webcam will be shown with a take picture button. When button pressed picture will be saved in local folder and image will be processed and shown at /result directory. Result image will be destroyed on back to home button clicked.

## This is an intern test module for PT Kazee Digital Indonesia, Waktoo Product. Currently on Pre-Alpha 2.
