import cv2 as cv
import numpy as np
import os
path = 'F:\PycharmProjects\monkey2\monkeyimages'
# intializing
orb = cv.ORB_create(nfeatures=1000)

images = [] # stores the images
classnames = [] # stores names of images
mylist = os.listdir(path)
print(mylist) # displays the names of images
print('Total classes Detected: ',len(mylist))

# importing multiple images loop is considered
# Displaying image name on object
for cl in mylist:
    imgcur = cv.imread(f'{path}/{cl}',0)
    images.append(imgcur) # adding images to a list
    # excluding extension of image
    classnames.append(os.path.splitext(cl)[0])
# print(classnames)

# storing images descriptors
def findDes(images):
    desList=[] # stores images des in desList
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

# comparing des from frame and image
def findID(img,desList,thres=20):
    # frame descriptor
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv.BFMatcher()
    matchList=[]
    finalVal = -1 # index of images
    try:
     for des in desList:
      matches = bf.knnMatch(des,des2,k=2)
      good = []
      for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
      # print(len(good))
      matchList.append(len(good))
    except:
     pass
    # print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            # print(max(matchList)) ################### define relay function here #################
            finalVal = matchList.index(max(matchList))
    return finalVal


desList = findDes(images)
# print(len(desList))

# WEBcam
cap = cv.VideoCapture(0)
while True:
    success, frame = cap.read()
    imgOriginal=frame.copy()
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    id = findID(frame,desList)
    if id != -1:
        cv.putText(frame,classnames[id],(50,50),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
    cv.imshow('WEBCOLOR',frame)
    # cv.imshow('WEB',frame)
    cv.waitKey(1)







