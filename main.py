from random import sample
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

#Function for converting image_matrix(rgb) to binary matrix
def convert_img_to_binary(img_mat,reverse=False):
    for i in range(len(img_mat)):
        for j in range(len(img_mat[0])):
            if not reverse:
                # print(img_mat[i][j])
                if img_mat[i][j]>120:
                    img_mat[i][j] = 1
                elif img_mat[i][j]<120:
                    img_mat[i][j] = 0

            else:
                if img_mat[i][j]==1:
                    img_mat[i][j] = 255
                elif img_mat[i][j]==0:
                    img_mat[i][j] =  0        

#Function for flipping the binary matrixes
def flipping(img,probability):
    convert_img_to_binary(img)
    total = img.shape[0] * img.shape[1]
    no_of_flips = round(total*probability)
    for i in range(no_of_flips):
        x = np.random.choice(np.arange(img.shape[0]))
        y = np.random.choice(np.arange(img.shape[1]))
        if img[x][y] == 0:
            img[x][y] =1
        elif img[x][y] == 1:
            img[x][y] = 0 
    convert_img_to_binary(img,reverse=True)
    return img

#Redundant transmission of image for reducing the noisy element
def recover(img,prob,no_redun):
    tests = []
    for i in range(no_redun):
        noise = np.array(img)
        flipping(noise,prob)
        # print(noise==org_img)
        distort_img = Image.fromarray(noise)
        # distort_img.show()
        tests.append(noise)
    img_recover =np.array([[ 0 for _ in range(org_img.shape[1])]for _ in range(org_img.shape[0])])
    # print(img_recover.shape)
    for i in range(len(org_img)):
        for j in range(len(org_img[0])):
            pix = []
            for k in range(no_redun):
                    pix.append(tests[k][i][j])
            img_recover[i][j] = mode(pix)

#     # print(img_recover.shape)
#     # print(img_recover)
    return img_recover


img = Image.open('./Images/dogo.jpg').convert('L')          #IMAGE OPENING HERE!!

org_img = np.array(img)
print(org_img.shape)
convert_img_to_binary(org_img)
convert_img_to_binary(org_img,reverse=True)

sample = Image.fromarray(org_img)
# sample.show()

probabilities = [0.05,0.15,0.25,0.5]                        #Probability list

#Subplots
plt.subplot(3,3,2)                                  
plt.title("Original Image")
plt.xticks([])
plt.yticks([])
plt.imshow(sample)


for i in range(4,8):
    test = org_img
    flipping(test,probability=probabilities[i-4])
    distort_img = Image.fromarray(test) 
    distort_img.save(f"test_img{i-4}.jpg")

    plt.subplot(3,3,i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test)
    plt.title(f"Distorted image #{i-3}")
plt.show()


#Recovering 
for i in range(4):
    for j in range(3,10,2):
        a = recover(img,probabilities[i],j)
        # print(a==org_img)
        pic = Image.fromarray((a).astype(np.uint8))
        # pic.show()
        pic.save(f"./results/ant_recover{i+1}.{j}.jpg")








