#import All libraries
import time
import os
import cv2
import argparse
import imutils
from imutils.video import VideoStream

#argparse function
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())

#detector har cascade for face detection
detector = cv2.CascadeClassifier(args["cascade"])

print("[INFO] starting video stream")
vs = VideoStream(src=0).start()

time.sleep(2.0)
total = 0

while True:
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	#detect the face in graysacle
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30))
	#loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	#show the output frame

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("k"):#here you press the k button then you pic will store given dir
		p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	elif key == ord("q"):#terminate the open window
		break

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up....")
cv2.destroyAllWindows()
vs.stop()		
		












