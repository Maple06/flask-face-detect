# Face Recognition Website using Flask and OpenCV.
## Features
1. Live webcam
2. Picture taking
3. Face detection from recent taken picture

## How To Use
### !! THIS CURRENT COMMIT HAS NOT BEEN TESTED FOR DOCKER YET !!
Build a docker image by running `docker build -t <image-name> .`, then create a container with `docker run --name <container-name> -p 8000:8000`. After running the container, go to localhost:8000. On the landing page ("/" directory), a webcam will be shown with a "Take Picture" button. When the button is pressed, the picture will be saved in a local folder, and the image will be processed and shown at "/result" directory. Result image will be destroyed on "Back to home" button clicked.

## This is an intern test module for PT Kazee Digital Indonesia, Waktoo Product. Currently on Pre-Alpha 2.
