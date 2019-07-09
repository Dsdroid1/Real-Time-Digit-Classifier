import cv2
import numpy as np

#'c' To clear screen
#'SPACE' to predict

l_h=0
l_s=0
l_v=0
u_h=255
u_s=255
u_v=255
pt_list=list()
num = np.zeros((480,640,3), np.uint8)+255#From webcam size


def nothing(x):#dummy fn
    pass


def calibrate_stylus(Mode):
    global l_h,l_s,l_v,u_h,u_s,u_v
    if Mode=='Calibrate':
        cap = cv2.VideoCapture(0);

        cv2.namedWindow('Thresh')
        cv2.createTrackbar("LH", 'Thresh', 0, 255, nothing)
        cv2.createTrackbar("LS", 'Thresh', 0, 255, nothing)
        cv2.createTrackbar("LV", 'Thresh', 0, 255, nothing)
        cv2.createTrackbar("UH", 'Thresh', 255, 255, nothing)
        cv2.createTrackbar("US", 'Thresh', 255, 255, nothing)
        cv2.createTrackbar("UV", 'Thresh', 255, 255, nothing)

        while True:
            _,frame=cap.read()
            frame = cv2.flip(frame, 1)
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)#To convert to HSV for masking

            l_h = cv2.getTrackbarPos('LH', 'Thresh')
            l_s = cv2.getTrackbarPos('LS', 'Thresh')
            l_v = cv2.getTrackbarPos('LV', 'Thresh')
            u_h = cv2.getTrackbarPos('UH', 'Thresh')
            u_s = cv2.getTrackbarPos('US', 'Thresh')
            u_v = cv2.getTrackbarPos('UV', 'Thresh')

            lower_limit=np.array([l_h,l_s,l_v])
            upper_limit=np.array([u_h,u_s,u_v])

            mask = cv2.inRange(hsv, lower_limit, upper_limit)
            result=cv2.bitwise_and(frame,frame,mask=mask)
            cv2.imshow("Mask",mask)
            cv2.imshow("Result",result)
            cv2.imshow('Frame',frame)

            # To finalise values,press esc
            key = cv2.waitKey(1)
            if key == 27:
                values = open('values.txt', 'w')
                values.write(str(l_h)+'\n')
                values.write(str(l_s)+'\n')
                values.write(str(l_v)+'\n')
                values.write(str(u_h)+'\n')
                values.write(str(u_s)+'\n')
                values.write(str(u_v)+'\n')
                break
        cap.release()
        cv2.destroyAllWindows()


    elif Mode=='Read':
        f = open("values.txt", 'r')
        value = list()
        for line in f:
            value.append(int(line))

        l_h=value[0]
        l_s=value[1]
        l_v=value[2]
        u_h=value[3]
        u_s=value[4]
        u_v=value[5]



def find_centroid(image):
    x_cm=0
    y_cm=0
    image=255-image#finding cenroid for the number part,not the background
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
    img2=image[:,:,0]/3+image[:,:,1]/3+image[:,:,2]/3
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

import torch
import torch.nn as nn
import torch.nn.functional as F





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


def Video_draw(mode):
    global l_h, l_s, l_v, u_h, u_s, u_v,num
    pred1=None
    cap = cv2.VideoCapture(0);

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        lower_limit=np.array([l_h,l_s,l_v])
        upper_limit=np.array([u_h,u_s,u_v])
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
        # drawing contour
        n = 0 #no.of contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 600 and area < 3000:
                n += 1
                M = cv2.moments(contour)

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
                pt_list.append((cx, cy))

        for i in range(len(pt_list) - 1):
            cv2.line(frame, pt_list[i], pt_list[i + 1], (0, 255, 0), 5)
            cv2.line(num, pt_list[i], pt_list[i + 1], (0, 0, 0), 15)



        if n == 0 and mode=='auto' and np.max(num)!=np.min(num):


            cv2.imwrite("Image.jpeg", num)
            img = cv2.imread("Image.jpeg")

            i = preprocess(img)
            x_i, y_i = find_centroid(i)
            # print("Current centroid:" + '(' + str(x_i) + ',' + str(y_i) + ')')
            i = padder(i, x_i, y_i)
            x_f, y_f = find_centroid(i)
            # print("New centroid:" + '(' + str(x_f) + ',' + str(y_f) + ')')

            input = torch.from_numpy(255 - i)
            pred = net(input.unsqueeze(0).unsqueeze(0).float())
            # print("Prediction:" + str(int(pred.argmax(dim=1))))
            pred1 = int(pred.argmax(dim=1))
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Prediction:" + str(pred1)
            cv2.putText(frame, text, (40, 40), font, 1, (183, 234, 255), 3)
            cv2.imshow("Frame", frame)

        if mode=='auto' and n==0:
            pt_list.clear()
            num = np.zeros((480, 640, 3), np.uint8) + 255



        #cv2.imshow("Frame",frame)
        #cv2.imshow("Image",num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Prediction:" + str(pred1)
        cv2.putText(frame,text,(40,40),font,1,(183,234,255),3)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key==27:
            break
        if key == 99 and mode=='manual':#'c' To clear the pt. list
            pt_list.clear()
            num = np.zeros((480, 640, 3), np.uint8) + 255
        if key == 32 and mode=='manual':#space
            cv2.imwrite("Image.jpeg",num)
            img=cv2.imread("Image.jpeg")

            i=preprocess(img)
            x_i, y_i = find_centroid(i)
            #print("Current centroid:" + '(' + str(x_i) + ',' + str(y_i) + ')')
            i = padder(i, x_i, y_i)
            x_f, y_f = find_centroid(i)
            #print("New centroid:" + '(' + str(x_f) + ',' + str(y_f) + ')')

            input = torch.from_numpy(255 - i)
            pred = net(input.unsqueeze(0).unsqueeze(0).float())
            #print("Prediction:" + str(int(pred.argmax(dim=1))))
            pred1=int(pred.argmax(dim=1))


    cap.release()
    cv2.destroyAllWindows()


#calibrate_stylus("Calibrate")
calibrate_stylus("Read")
Video_draw("auto")