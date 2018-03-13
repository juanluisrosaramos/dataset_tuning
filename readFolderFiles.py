import os
import random
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import uuid
from subprocess import call


'''
Function to obtain a list of files from a directory send by arguments
(without final /)
'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
                help="Needs a folder where the images are")
ap.add_argument("-t", "--task", required=True,
                help="Needs task align or augment")
ap.add_argument("-q", "--q", required=False,
                help="Needs quantity of images to augment")

args = vars(ap.parse_args())


def list_files(dir):
    r = []
    r_subdir = []
    r_file = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        # files = os.walk(subdir).__next__()[2]
        files = next(os.walk(subdir))[2]
        if (len(files) > 0):
            for file in files:
                r.append(subdir + "/" + file)
                r_subdir.append(subdir)
                r_file.append(file)
    return r, r_subdir, r_file


def align_faces(folder):
    # call align faces with each file of the folder
    #folder='/home/monete/monete@gmail.com/studying/IA/thesis/deeplearning/dataset/fer2013/PublicTest/0-neutral'
    output_folder = folder+'/output/'
    print('********************\nBegin')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shape_predictor = '/home/monete/monete@gmail.com/coding/MAI/thesis/dataset_tuning/shape_predictor_68_face_landmarks.dat'
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    # 0.25 is the desired zoom 0.25 is the default
    fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25),desiredFaceWidth=224)
    # Folder to read
    r_f, r_subdir_f, r_file_f_pre = list_files(folder)
    for i, value in enumerate(r_f, 0):
        #Image = r_f[i]
        print r_f[i]
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(r_f[i])
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image
        rects = detector(gray, 0)
        print ('RECTS: ' + str(len(rects)))
        for rect in rects:
        	# extract the ROI of the *original* face, then align the face
        	# using facial landmarks
            print('********************\nStart ' + str(i) + ' of ' + str(len(r_f)))
            (x, y, w, h) = rect_to_bb(rect)
            faceAligned = fa.align(image, gray, rect)
            f = str(uuid.uuid4())
        # write resulting image
            #print output_folder + f + '.png'
            cv2.imwrite(output_folder + f + ".png", faceAligned)
            print('********************\nEnd ' + str(i)  + ' of ' + str(len(r_f)))
    print('********************\nTOTAL End')
    os.system('rm ' + folder + '/*')
    os.system('mv ' + folder + '/output/* ' + folder)
    print ('rm -R ' + folder + '/output')
    #os.system('rm -R ' + folder + ' /output')

def augment_files(folder, q):
    os.system('mkdir out')
    # select actions
    print('********************\nBegin augment dataset')

    actions = ['fliph,blur_1.5,rot_-2', 'fliph,blur_0.5,rot_4,noise_0.009',
               'fliph,rot_-3,noise_0.0009',
               'fliph,rot_2,blur_1,noise_0.0012',
               'fliph,rot_-2,blur_1.2,noise_0.0012']
    for i in range(int(q)):
        # Select a random file from Folder and an action
        action = random.choice(actions)
        file = random.choice(os.listdir(folder))
        os.system('python ~/monete@gmail.com/coding/MAI/thesis/dataset_tuning/image_augmentor/main.py ' + folder + ' '+file + ' ' + action)
    # Select a random action
    # os.system('mv out/* .')
    # os.system('rm -R out/')

# ############################################

# parsing arguments


task = args["task"]
folder = args["folder"]
quantity = args["q"]

if task == 'augment':
    augment_files(folder, quantity)
    print('Augmenting dataset')
else:
    align_faces(folder)
    print('Aligning faces')

#############################################
