#Digit Recognition

import cv2
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1
line_thick=20

def num_draw(event,x,y,flags,param):  #flags and param are some arguments req. for the setMouseCallBack
    global ix, iy, drawing, mode,line_thick

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img,(ix,iy),(x,y),(0,0,0),line_thick)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 0, 0), line_thick)

img = np.zeros((512,512,3), np.uint8)+255
cv2.namedWindow('image')
cv2.setMouseCallback('image',num_draw)


def start_draw():  #Opens the drawing screen
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    #name=input("Name of img?")
    #cv2.imwrite(name+".jpeg",img)
    cv2.destroyAllWindows()

#print(img.shape)
def find_centroid(image):
    x_cm=0
    y_cm=0
    image=255-image
    for i in range(1,image.shape[0]+1):
        y_cm+=np.sum(i*image[:,i-1])
    for i in range(1,image.shape[1]+1):
        x_cm+=np.sum(i*image[i-1,:])
    y_cm=int(y_cm/np.sum(image))
    x_cm=int(x_cm/np.sum(image))
    return x_cm,y_cm


def padder(image, x, y):

    k = 10 - y
    h = 10 - x
    #Importatnt:The padding may seem a bit odd as if the centroid is at 12(instead of 10),then shouldn't
    #we pad more on the right(towards 28) to get to 14....
    #But.....
    #White pixel has value 255,and black zero
    #We are inverting things hence the centroid is in a different img form(Check the 255-img)
    #This is why padding is inverted
    #This code is not robust...i.e will give error if 4-h or something becomes -ve..../
    #Try to improve that
    #By this preprocessing 6,7,9 can be identified..(which earlier was giving 8,2,8)
    image = np.pad(np.array(image), ((4 + h, 4 - h), (4 + k, 4 - k)), 'constant', constant_values=(255, 255))
    return image


def preprocess(image):
    img2=img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3
    v_s = 0
    v_e = img2.shape[0]

    h_s = 0
    h_e = img2.shape[1]

    for i in range(img2.shape[0]):
        if np.min(img2[i, :]) > 30:
            v_s += 1
        else:
            break
    for i in reversed(range(img2.shape[0])):
        if np.min(img2[i, :]) > 30:
            v_e -= 1
        else:
            break
    for i in range(img2.shape[1]):
        if np.min(img2[:, i]) > 30:
            h_s += 1
        else:
            break
    for i in reversed(range(img2.shape[1])):
        if np.min(img2[:, i]) > 30:
            h_e -= 1
        else:
            break

    # Slicing that image
    img3 = img2[v_s:v_e, h_s:h_e]
    # Maintaining the aspect ratio
    ih = h_e - h_s
    iv = v_e - v_s
    if ih > iv:
        p = int((ih - iv) / 2)
        img4 = np.pad(np.array(img3), ((p, p), (0, 0)), 'constant', constant_values=(255, 255))
    else:
        p = int((iv - ih) / 2)
        img4 = np.pad(np.array(img3), ((0, 0), (p, p)), 'constant', constant_values=(255, 255))

    img5 = cv2.resize(img4, (20, 20))
    #img6 = np.pad(img5, ((4, 4), (4, 4)), 'constant', constant_values=(255, 255))

    return img5


start_draw()
i=preprocess(img)
x_i,y_i=find_centroid(i)
print("Current centroid:"+'('+str(x_i)+','+str(y_i)+')')
i=padder(i,x_i,y_i)
x_f,y_f=find_centroid(i)
print("New centroid:"+'('+str(x_f)+','+str(y_f)+')')

cv2.imshow('Number', i)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(i.shape)

#cv2.imwrite('i.jpeg',i)

import torch
import torch.nn as nn
import torch.nn.functional as F


#This converts numpy to tensor
input=torch.from_numpy(255-i)
#input.unsqueeze(0)
#print(input.unsqueeze(0).unsqueeze(0).shape)


class Network(nn.Module):  # Extends all features of Module(bulit in class)to oyr network class
    def __init__(self):  # Defines the type of layers as part of the object
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5, stride=(1, 1))

        self.fc1 = nn.Linear(in_features=10 * 4 * 4, out_features=90)
        self.out = nn.Linear(in_features=90, out_features=10)

    def forward(self, t):
        # Conv layer 1
        t = self.conv1(t)  # Implements the forward pass through predefined functions in the nn.Module class
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Conv layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Fc layer 1
        t = t.reshape(-1, 10 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # Final softmax layer
        t = self.out(t)
        # t=F.softmax(t,dim=1),No need to do this cross entropy does it by itself

        return t

net=torch.load('ConvNet MNIST.pth')
net.eval()

pred=net(input.unsqueeze(0).unsqueeze(0).float())
#print(pred)

print("Prediction:"+str(int(pred.argmax(dim=1))))


