# This script imports a .csv file containing filepath to images. The images are read, augmented and writing new augmented images as a dat file. 


import random
import imageio
import imgaug as ia
import numpy as np
import matplotlib.pyplot as plt
import csv
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from os import listdir
from os.path import isfile, join

#CONFIG
i=0
ia.seed(360)
variationsOfEachImage = 19


folderOfImagesToBeAugmented = "CameraSet1/Images"
folderOfAugmentedImages     = "AugmentedSet1"
gTruthFileName              = "cameraset1.csv"
imageFilePathInMatlab       = "C:/Users/toran/OneDrive - Universitetet i Agder/MAS500-Swarm/AI/TrainingData/AugmentedSet7/"


gTruthPath = folderOfAugmentedImages + '/' + gTruthFileName

# https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
# #Set up augmentations
seq = iaa.Sequential([
    
    iaa.Sometimes(0.3,
        iaa.OneOf([ 
           iaa.MotionBlur(k=(5,10), angle=(60, 120)),
           iaa.imgcorruptlike.DefocusBlur(severity=(1,2)),
           # iaa.imgcorruptlike.ZoomBlur(severity=1),
           iaa.GaussianBlur(sigma=(0.3, 0.5))
     ])
    ),

    iaa.Sometimes(0.05,
             iaa.AveragePooling(1)
    ),

    # iaa.Sometimes(0.05,
    #     iaa.OneOf([
    #         iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.05)),
    #         iaa.Cutout(nb_iterations=(1, 3), size=(0.05, 0.15), fill_mode="constant", cval=(0, 255))
    #     ]),
    # ),

    iaa.Fliplr(0.5),

    # iaa.Sometimes(0.1,
    #     iaa.Crop(percent=(0.05, 0.15))
    # ),
    
    iaa.OneOf([
        iaa.LinearContrast((0.80, 1.1)),
        iaa.Multiply((0.9, 1.5)),
        # iaa.pillike.Autocontrast((10, 20), per_channel=True),
        iaa.Sometimes(0.50,
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255)),
                iaa.PiecewiseAffine(scale=(0.01, 0.05))
                ]),
        ),
    ]),

    iaa.Sometimes(0.8,
        iaa.OneOf([
            iaa.OneOf([
                iaa.ShearX((-5,5)),
                iaa.ShearY((-5,5))
            ]),
            iaa.OneOf([
                iaa.Affine(
                    scale={"x": (0.8, 1), "y": (0.8, 1)}
                ),
                iaa.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
                ),
                iaa.Affine(
                    rotate=(-4, 4)
                )
            ]),
        ]),
    ),
], random_order=True) # apply augmenters in random order





#Run Augmenter

with open(gTruthPath) as csvfile:
    txtresultfile = open(folderOfAugmentedImages + '/gTruth' + folderOfAugmentedImages + 'Table.dat', 'w', newline='\n')

    #Get image name and boundingbox data from table
    readCSV = csv.reader(csvfile, delimiter=',')

    txtresultfile.writelines('imageFilePath'+'|'+'Loomo'+ '\n')
    for row in readCSV:
        #Load image
        fullPath = folderOfImagesToBeAugmented+"/"+str(row[0])
        print(fullPath)
        image = imageio.imread(fullPath)
        #ia.imshow(image)
        #Load bounding box data and save in Int list
        bbstring = row[1].replace('[','').replace(']','').replace(';',',').split(',')
        bbCoord = list(map(int, map(float, bbstring)))
        numberOfBB = len(bbstring)/4.0
        # print(numberOfBB)
        # print(bbCoord)
        
        #Apply bounding box to images
        if(numberOfBB == 1.0):
            # print('1 Bounding box')
            bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbCoord[0], x2=bbCoord[0]+bbCoord[2], y1=bbCoord[1], y2=bbCoord[1]+bbCoord[3])
            ], shape=image.shape)
        if(numberOfBB == 2.0):
            # print('2 Bounding box')
            bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbCoord[0], x2=bbCoord[0]+bbCoord[2], y1=bbCoord[1], y2=bbCoord[1]+bbCoord[3]),
            BoundingBox(x1=bbCoord[4], x2=bbCoord[4]+bbCoord[6], y1=bbCoord[5], y2=bbCoord[5]+bbCoord[7])
            ], shape=image.shape)

        #Save original image
        augmentedpath = folderOfAugmentedImages + '/'+ folderOfAugmentedImages + '_img_'+str(i)+'.png'
        imageio.imwrite(augmentedpath, image[:, :])

        #Save Bounding box string
        if(numberOfBB == 1.0):
            # print(bbs_aug[0])
            newbbstring = '['+str(int(bbs[0][0][0]))+' '+str(int(bbs[0][0][1]))+' '+str(int(bbs[0][1][0]-bbs[0][0][0]))+' '+str(int(bbs[0][1][1]-bbs[0][0][1]))+']'
        if(numberOfBB == 2.0):
            # print(bbs_aug[0])
            # print(bbs_aug[1])
            newbbstring = '['+str(int(bbs[0][0][0]))+' '+str(int(bbs[0][0][1]))+' '+str(int(bbs[0][1][0]-bbs[0][0][0]))+' '+str(int(bbs[0][1][1]-bbs[0][0][1]))+';'+str(int(bbs[1][0][0]))+' '+str(int(bbs[1][0][1]))+' '+str(int(bbs[1][1][0]-bbs[1][0][0]))+' '+str(int(bbs[1][1][1]-bbs[1][0][1]))+']'
        # print(newbbstring)

        imagepath = imageFilePathInMatlab + folderOfAugmentedImages + '_img_'+str(i)+'.png'
        txtresultfile.writelines(imagepath+'|'+newbbstring+ '\n')

        i = i+1

        for x in range(variationsOfEachImage):
            
            #Perform augmentation
            image_aug,bbs_aug = seq(image=image, bounding_boxes=bbs)

            #Save image
            augmentedpath = folderOfAugmentedImages + '/'+ folderOfAugmentedImages + '_img_'+str(i)+'.png'
            imageio.imwrite(augmentedpath, image_aug[:, :])
            
            #Save Bounding box string
            if(numberOfBB == 1.0):
                # print(bbs_aug[0])
                newbbstring = '['+str(int(bbs_aug[0][0][0]))+' '+str(int(bbs_aug[0][0][1]))+' '+str(int(bbs_aug[0][1][0]-bbs_aug[0][0][0]))+' '+str(int(bbs_aug[0][1][1]-bbs_aug[0][0][1]))+']'
            if(numberOfBB == 2.0):
                # print(bbs_aug[0])
                # print(bbs_aug[1])
                newbbstring = '['+str(int(bbs_aug[0][0][0]))+' '+str(int(bbs_aug[0][0][1]))+' '+str(int(bbs_aug[0][1][0]-bbs_aug[0][0][0]))+' '+str(int(bbs_aug[0][1][1]-bbs_aug[0][0][1]))+';'+str(int(bbs_aug[1][0][0]))+' '+str(int(bbs_aug[1][0][1]))+' '+str(int(bbs_aug[1][1][0]-bbs_aug[1][0][0]))+' '+str(int(bbs_aug[1][1][1]-bbs_aug[1][0][1]))+']'
            # print(newbbstring)

            # Save table
            imagepath = imageFilePathInMatlab + folderOfAugmentedImages + '_img_'+str(i)+'.png'
            txtresultfile.writelines(imagepath+'|'+newbbstring+ '\n')

            i = i+1 
