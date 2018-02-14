import os
import random
import argparse
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
    print('********************\nBegin')
    # Folder to read
    r_f, r_subdir_f, r_file_f_pre = list_files(folder)
    print r_f
    from subprocess import call
    for i, value in enumerate(r_f, 1):
        print r_f[i]
        z = ' --shape-predictor shape_predictor_68_face_landmarks.dat --image '
        + r_f[i]
        print z
        print i
        # call align faces file
        # os.system('python align_faces.py' + z)
        print(i)
    print('********************\nEnd')


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
