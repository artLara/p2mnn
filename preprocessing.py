import mediapipe as mp
import pandas as pd
import cv2
import json 
import os 
import numpy as np


mp_holistic = mp.solutions.holistic
# def writeInformation(frames, landmarks):
      # path2write = path + 'images/' + videoName
      #    if not os.path.exists(path2write+videoName): 
      #       os.makedirs(path2write+videoName) 
#    saveImage(image, imageName=videoName+'_'+ind, path=path2write+'/')
   
# def get_landmarks_from_video(videoName, path=''): Need too much RAM
#    frames = get_frames(videoName=videoName, path=path)
#    print('Do it')
#    landmarks = get_landmark_from_images(frames)
   # writeInformation()

def padding_zeros(image, width = 300, height = 300, channels = 3, color = None):
   """
   Description: add padding of zeros an numpy array (image)
   """
   if color == None:
      color = (0,0,0)

   (h, w) = image.shape[:2]
   image_pad = np.full((height,width,channels), color, dtype=np.uint8)
   offsetHeight1 = (height-h)//2
   offsetHeight2 = (height-h)//2 + h
   offsetWidth1 = (width-w)//2
   offsetWidth2 = (width-w)//2 + w
   image_pad[offsetHeight1:offsetHeight2, offsetWidth1:offsetWidth2] = image
   
   return image_pad


def image_resize(image, width = 300, height = 300, inter = cv2.INTER_AREA):
   """
   Description: resize an image saving aspect realtion
   """
   (h, w) = image.shape[:2]
   #Case 1: height is longer than width
   if h > w:
      relationAspect = height / float(h)
      dim = (int(w * relationAspect), height)

   else:
      relationAspect = width / float(w)
      dim = (width, int(h * relationAspect))

   resized = cv2.resize(image, dim, interpolation = inter)
   resized = padding_zeros(resized)
   return resized

def get_frames(videoName, path=''):
   """
   Description: This function returns a list of frames of video
   """
   frames = []
   vidcap = cv2.VideoCapture(path + 'all_videos/' + videoName)
   successImage,image = vidcap.read()
   while successImage:
      frames.append(image)
      successImage,image = vidcap.read()
      # print('Frame '.forma)

   return frames

# def get_landmark_from_images(images):
#    """
#    Description: This function returns landmarks from an images list
#    """
#    landmarks = []
#    with mp_holistic.Holistic(
#       static_image_mode=True,
#       model_complexity=2,
#       enable_segmentation=True,
#       refine_face_landmarks=True) as holistic:
#       for image in images:
#       # image = cv2.imread(file)
#          image_height, image_width, _ = image.shape
#          # Convert the BGR image to RGB before processing.
#          results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#          if results.pose_landmarks:
#             print(
#                f'Nose coordinates: ('
#                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#             )
#             landmarks.append(results)
#             print(results.pose_landmarks.landmark)
#          else:
#             landmarks.append(None)

#    return landmarks 

# def saveLandmarks(landmarks_list, image_shape, name, path):
#    """
#    Description: save landmarks in specific path
#    """
#    landmarks_dict = {}
#    for ind, landmarks in enumerate(landmarks_list):
#       landmarks_dict['frame'+str(ind)] = {'NOSE':{'x':None, 'y':None, 'z':None, 'indexMP':0},
#                                           'LEFT_EYE_INNER':{'x':None, 'y':None, 'z':None, 'indexMP':1},
#                                           'LEFT_EYE':{'x':None, 'y':None, 'z':None, 'indexMP':2},
#                                           'LEFT_EYE_OUTER':{'x':None, 'y':None, 'z':None, 'indexMP':3},
#                                           'RIGHT_EYE_INNER':{'x':None, 'y':None, 'z':None, 'indexMP':4},
#                                           'RIGHT_EYE':{'x':None, 'y':None, 'z':None, 'indexMP':5},
#                                           'RIGHT_EYE_OUTER':{'x':None, 'y':None, 'z':None, 'indexMP':6},
#                                           'LEFT_EAR':{'x':None, 'y':None, 'z':None, 'indexMP':7},
#                                           'RIGHT_EAR':{'x':None, 'y':None, 'z':None, 'indexMP':8},
#                                           'MOUTH_LEFT':{'x':None, 'y':None, 'z':None, 'indexMP':9},
#                                           'MOUTH_RIGHT':{'x':None, 'y':None, 'z':None, 'indexMP':10},
#                                           'LEFT_SHOULDER':{'x':None, 'y':None, 'z':None, 'indexMP':11},
#                                           'RIGHT_SHOULDER':{'x':None, 'y':None, 'z':None, 'indexMP':12},
#                                           'LEFT_ELBOW':{'x':None, 'y':None, 'z':None, 'indexMP':13},
#                                           'RIGHT_ELBOW':{'x':None, 'y':None, 'z':None, 'indexMP':14},
#                                           'LEFT_WRIST':{'x':None, 'y':None, 'z':None, 'indexMP':15},
#                                           'RIGHT_WRIST':{'x':None, 'y':None, 'z':None, 'indexMP':16},
#                                           'LEFT_PINKY':{'x':None, 'y':None, 'z':None, 'indexMP':17},
#                                           'RIGHT_PINKY':{'x':None, 'y':None, 'z':None, 'indexMP':18},
#                                           'LEFT_INDEX':{'x':None, 'y':None, 'z':None, 'indexMP':19},
#                                           'RIGHT_INDEX':{'x':None, 'y':None, 'z':None, 'indexMP':20},
#                                           'LEFT_THUMB':{'x':None, 'y':None, 'z':None, 'indexMP':21},
#                                           'RIGHT_THUMB':{'x':None, 'y':None, 'z':None, 'indexMP':22},
#                                           'LEFT_HIP':{'x':None, 'y':None, 'z':None, 'indexMP':23},
#                                           'RIGHT_HIP':{'x':None, 'y':None, 'z':None, 'indexMP':24},
#                                           'LEFT_KNEE':{'x':None, 'y':None, 'z':None, 'indexMP':25},
#                                           'RIGHT_KNEE':{'x':None, 'y':None, 'z':None, 'indexMP':26},
#                                           'LEFT_ANKLE':{'x':None, 'y':None, 'z':None, 'indexMP':27},
#                                           'RIGHT_ANKLE':{'x':None, 'y':None, 'z':None, 'indexMP':28},
#                                           'LEFT_HEEL':{'x':None, 'y':None, 'z':None, 'indexMP':29},
#                                           'RIGHT_HEEL':{'x':None, 'y':None, 'z':None, 'indexMP':30},
#                                           'LEFT_FOOT_INDEX':{'x':None, 'y':None, 'z':None, 'indexMP':31},
#                                           'RIGHT_FOOT_INDEX':{'x':None, 'y':None, 'z':None, 'indexMP':32}
#                                           }
      
#       for key in landmarks_dict['frame'+str(ind)].keys():
#          landmarks_dict['frame'+str(ind)][key]['x'] = landmarks.pose_landmarks.landmark[landmarks_dict['frame'+str(ind)][key]['indexMP']].x
#          landmarks_dict['frame'+str(ind)][key]['y'] = landmarks.pose_landmarks.landmark[landmarks_dict['frame'+str(ind)][key]['indexMP']].y
#          landmarks_dict['frame'+str(ind)][key]['z'] = landmarks.pose_landmarks.landmark[landmarks_dict['frame'+str(ind)][key]['indexMP']].z
      
#       landmarks_dict['img_height'] = image_shape[0]
#       landmarks_dict['img_width'] = image_shape[1]
#       landmarks_dict['img_channel'] = image_shape[2]
#    with open(path + name + ".json", "w") as outfile: 
#       json.dump(landmarks_dict, outfile)
def saveLandmarks(landmarks_list, image_shape, name, path, decimals = 4):
   """
   Description: save landmarks in specific path
   """
   landmarks_dict = {}
   m = 10**decimals
   for ind, landmarks in enumerate(landmarks_list):
      landmarks_dict['frame'+str(ind)] = {'pose':{'x':[], 'y':[]},
                                          'right_hand':{'x':[], 'y':[]},
                                          'left_hand':{'x':[], 'y':[]},
                                          'face':{'x':[], 'y':[]}
                                          # 'complete':False
                                          }
      
      complete = 0
      if landmarks.pose_landmarks:
         complete += 1
         for i in landmarks.pose_landmarks.landmark:
            landmarks_dict['frame'+str(ind)]['pose']['x'].append(int(i.x*m))
            landmarks_dict['frame'+str(ind)]['pose']['y'].append(int(i.y*m))
            # landmarks_dict['frame'+str(ind)]['pose']['z'].append(i.z)

      if landmarks.right_hand_landmarks:
         complete += 1
         for i in landmarks.right_hand_landmarks.landmark:
            landmarks_dict['frame'+str(ind)]['right_hand']['x'].append(int(i.x*m))
            landmarks_dict['frame'+str(ind)]['right_hand']['y'].append(int(i.y*m))
            # landmarks_dict['frame'+str(ind)]['right_hand']['z'].append(i.z)

      if landmarks.left_hand_landmarks:
         complete += 1
         for i in landmarks.left_hand_landmarks.landmark:
            landmarks_dict['frame'+str(ind)]['left_hand']['x'].append(int(i.x*m))
            landmarks_dict['frame'+str(ind)]['left_hand']['y'].append(int(i.y*m))
            # landmarks_dict['frame'+str(ind)]['left_hand']['z'].append(i.z)

      if landmarks.face_landmarks:
         complete += 1
         for i in landmarks.face_landmarks.landmark:
            landmarks_dict['frame'+str(ind)]['face']['x'].append(int(i.x*m))
            landmarks_dict['frame'+str(ind)]['face']['y'].append(int(i.y*m))
            # landmarks_dict['frame'+str(ind)]['face']['z'].append(i.z)

      # if complete == 4:
      #    landmarks_dict['frame'+str(ind)]['complete'] = True

      # for key in landmarks_dict['frame'+str(ind)].keys():
      #    landmarks_dict['frame'+str(ind)][key]['x'] = landmarks.pose_landmarks.landmark[landmarks_dict['frame'+str(ind)][key]['indexMP']].x
      #    landmarks_dict['frame'+str(ind)][key]['y'] = landmarks.pose_landmarks.landmark[landmarks_dict['frame'+str(ind)][key]['indexMP']].y
      #    landmarks_dict['frame'+str(ind)][key]['z'] = landmarks.pose_landmarks.landmark[landmarks_dict['frame'+str(ind)][key]['indexMP']].z
      
   landmarks_dict['img_height'] = image_shape[0]
   landmarks_dict['img_width'] = image_shape[1]
   landmarks_dict['img_channel'] = image_shape[2]
   with open(path + name + ".json", "w") as outfile: 
      json.dump(landmarks_dict, outfile)

def saveImage(image, imageName, path):
   """
   Description: save image in specific path
   """
   if not os.path.exists(path): 
      os.makedirs(path)
   cv2.imwrite(path + '/' + imageName + ".jpg", image)


def get_landmark_from_image(image):
   """
   Description: This function returns landmarks from an image
   """
   with mp_holistic.Holistic(
      static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True,
      refine_face_landmarks=True) as holistic:
      # image = cv2.imread(file)
      image_height, image_width, _ = image.shape
      # Convert the BGR image to RGB before processing.
      results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # if results.right_hand_landmarks:
         # print(
         #    f'Nose coordinates: ('
         #    f'{results.pose_landmarks.landmark[0].x * image_width}, '
         #    f'{results.pose_landmarks.landmark[0].y * image_height})'
         # )

         # print(
         #    f'LEFT_EYE_INNER coordinates: ('
         #    f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].x * image_width}, '
         #    f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].y * image_height})'
         # )
         # print(results.pose_landmarks)
         # print(results.right_hand_landmarks)
         # print(results.left_hand_landmarks)
         # print(results.face_landmarks)

         # return True, results
   return results

def get_landmarks_from_video(videoName, path=''):
   """
   Description: This function returns a list of landmarks
   """
    
   landmarks = []
   vidcap = cv2.VideoCapture(path + 'all_videos/' + videoName)
   successImage,image = vidcap.read()
   image = image_resize(image)
   shape = image.shape
   ind = 0
   while successImage:
      # print('--------------------Frame{}----------------'.format(ind))
      try:
         landmark = get_landmark_from_image(image)
         if landmark.pose_landmarks:
            landmarks.append(landmark)
            saveImage(image, imageName=videoName[:-4]+'_'+str(ind), path=path+'images/'+videoName[:-4])
            ind += 1
         successImage,image = vidcap.read()
         image = image_resize(image)
      
      except:
         print('Problems in frame of '+videoName)
      # if ind > 0:
      #    break
   # print(landmarks)
   
   saveLandmarks(landmarks,
                 shape, 
                 name=videoName[:-4], 
                 path=path+'landmarks/')


path_dataset = '/home/lara/Documents/p2mnn/row_dataset_new.csv'
# path = '/media/lara/Elements/LSMTV/all_videos/'
df = pd.read_csv(path_dataset)
for ind in df.index:
   if not df.at[ind, 'downloaded']:
      continue

   print('Processing {}/{}'.format(ind, len(df.index)))
   get_landmarks_from_video(df.at[ind, 'videoName'], path='/home/lara/Documents/p2mnn/')
   if ind > 0:
      break
   # break
