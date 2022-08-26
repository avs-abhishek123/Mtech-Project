from AugmentationLibrary.StaticTransforms.staticTransforms import StaticTransformDataGenerator as AugStatic
import PIL
import cv2
from PIL import Image

aug_list = ["Blur",
"CLAHE",
"ColorJitter",
"Downscale",
"Emboss",
"GaussNoise",
"GaussianBlur",
"GlassBlur",
"HueSaturationValue",
"ISONoise",
"MedianBlur",
"MotionBlur",
"MultiplicativeNoise",
"Posterize",
"Sharpen",
"Superpixels",
"ToGray",
"ToSepia",
"HorizontalFlip",
"OpticalDistortion",
"GridDistortion"]

path= "C:/Users/HP/Desktop/aug/Object_detect_aug/upload"

lst_of_image_names=[]
lst_of_image_names_with_path=[]

subdirs_list = []

for path, subdirs, files in os.walk(path):
    for filename in files:
        f = os.path.join(path, filename)
        lst_of_image_names_with_path.append(f)
    for subdir in subdirs:
        subdirs_list.append(os.path.join(path,subdir))

all_files_sub = []
sub_files = []
for i in subdirs_list:
    for path, _, files in os.walk(i):
        for filename in files:
            f = os.path.join(path, filename)
            sub_files.append(f)
    print(len(sub_files))
    all_files_sub.append(sub_files)
    sub_files = []


for AugType in aug_list:
    #AugType=input("Give the type of augmentation you want for all the images of the minority class : ")

    newMinorityDirectoryname=classname+AugType
    print("newMinorityDirectoryname",newMinorityDirectoryname)
    x = os.path.join(finalMinorityImageDir, newMinorityDirectoryname)
    #print(x)
    newAugmentedMinorityClasspath=  x
    print("newAugmentedMinorityClasspath",newAugmentedMinorityClasspath)

    if not os.path.exists(newAugmentedMinorityClasspath):
        print("THE DIRECTORY DONT EXIST SO MAKING A NEW ONE")
        os.mkdir(newAugmentedMinorityClasspath)

    elif not os.path.exists(newAugmentedMinorityClasspath):
        print("THE DIRECTORY EXIST")
        
        os.mkdir(newAugmentedMinorityClasspath+"1")

    elif not os.path.exists(newAugmentedMinorityClasspath+"1"):
        print("THE DIRECTORY EXIST")
        
        os.mkdir(newAugmentedMinorityClasspath+"2")

    else:
        print("THE DIRECTORY EXIST")
        annotation_path="/mc2/SaiAbhishek/Improve_Model_Accuracy_using_Aug_Approaches/Single_Static_Augmentation/chest_X_ray_pneumonia/Augmented_Dataset/COCO4Each/"+newMinorityDirectoryname+"3.json"
        
        os.mkdir(newAugmentedMinorityClasspath+"3")


    # add more static augmentations

    an_object = AugDataCreator(MajorityImage_Dir,MinorityImage_Dir,newAugmentedMinorityClasspath,AugType,minclass,majorityclass,minorityclass_Size,majorityclass_Size)

    newfilenameList=an_object.singleAugmented_imageCreator()
    print(newfilenameList)




img=cv2.imread("C:/Users/HP/Desktop/aug/Object_detect_aug/upload/British Innovator/British Innovator_0add1cb8_1.jpg")
image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

an_object= AugStatic(image,1)

# return number of sample, store them in the folder

print("AugStatic."+augmentationtype+"()")


augmented_image=eval("an_object."+augmentationtype+"()")


Image.fromarray(augmented_image).save("C:/Users/HP/Desktop/aug/Object_detect_aug/upload_aug"+augmentationtype+".jpg")

