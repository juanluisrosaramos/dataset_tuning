
# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# 0.25 is the desired zoom 0.25 is the default
fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25),desiredFaceWidth=112)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 2)

# loop over the face detections
# rect contains the bounding boxes
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	faceAligned = fa.align(image, gray, rect)
	#faceAligned = cv2.resize(faceAligned, (224, 224))
	import uuid
	f = str(uuid.uuid4())
	# write resulting image
	cv2.imwrite("/home/monete/monete@gmail.com/studying/IA/thesis/deeplearning/dataset/fer2013/output/7-surprise/" + f + ".png", faceAligned)

	# display the output images
	#cv2.imshow("Aligned", faceAligned)
	#cv2.waitKey(0)
