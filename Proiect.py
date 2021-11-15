#%%
#%%
import cv2 as cv
import numpy as np
import math

f=open(r"C:\Users\Costangy\Desktop\an3sem1\analiza metanumerica\demo.txt","w")
g=open(r"C:\Users\Costangy\Desktop\an3sem1\analiza metanumerica\bonus.txt","w")
template=[]
path1=r"C:\Users\Costangy\Desktop\an3sem1\analiza metanumerica\\"
for i in range(1,10):
    template.append(cv.imread(path1 + str(i) + ".jpg",0))
    #template[i-1]=cv.normalize(template[i-1], None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    

def show_image(title,image):
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def preprocess_image(image):
        img=image
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image_m_blur = cv.medianBlur(image,5)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
        image_sharpened = cv.addWeighted(image, 1, image_g_blur, -0.81, 0)
        _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)
    
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.erode(thresh, kernel)
        
        #show_image("median blurred",image_m_blur)
        #show_image("gaussian blurred",image_g_blur)
        #show_image("sharpened",image_sharpened)    
        show_image("threshold of blur",thresh)
        
        edges =  cv.Canny(thresh ,100,400)
        contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
       
        
        for i in range(len(contours)):
            if(len(contours[i]) >3):
                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point
    
                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                        possible_bottom_right = point
    
                diff = np.diff(contours[i].squeeze(), axis = 1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left
                    con=i
        cnt=contours[con]
        rect = cv.minAreaRect(cnt)  
        box = cv.boxPoints(rect) 
        box = np.int0(box)
        
        W = rect[1][0]
        H = rect[1][1]
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        angle = rect[2]   
        print(angle)
        if angle>15:
            angle+=270
# Center of rectangle in source image
        center = ((x1+x2)/2,(y1+y2)/2)
# Size of the upright rectangle bounding the rotated rectangle
        size = (x2-x1, y2-y1)
        M = cv.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
# Cropped upright rectangle
        cropped = cv.getRectSubPix(img, size, center)
        cropped = cv.warpAffine(cropped, M, size)
        croppedW = H if H > W else W
        croppedH = H if H < W else W
# Final cropped & rotated rectangle
        croppedRotated = cv.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
        show_image("croppedRotated", croppedRotated[:,13:-13])
        
    
        width = 500
        height = 500
   
        image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)
        cv.circle(image_copy,tuple(top_left),4,(0,0,255),-1)
        cv.circle(image_copy,tuple(top_right),4,(0,0,255),-1)
        cv.circle(image_copy,tuple(bottom_left),4,(0,0,255),-1)
        cv.circle(image_copy,tuple(bottom_right),4,(0,0,255),-1)
        return croppedRotated[:,13:-13]
        #show_image("detected corners",image_copy)
        #return top_left,top_right,bottom_left,bottom_right
def get_results(img,lines_horizontal,lines_vertical):
        #k=0
        for i in range(len(lines_horizontal)-1):
            for j in range(len(lines_vertical)-1):
                y_min = lines_vertical[j][0][0]
                y_max = lines_vertical[j + 1][1][0]
                x_min = lines_horizontal[i][0][1]
                x_max = lines_horizontal[i + 1][1][1]
                patch = img_crop[x_min:x_max, y_min:y_max].copy()
                
               # w, h = template[4].shape[::-1]
                img_gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                threshold = 0.9
 
                # Store the coordinates of matched area in a numpy array
                # Draw a rectangle around the matched region.
                # Show the final image with the matched area.
                #cv.imshow('Detected',patch)
                patch_2=img_crop[x_min+17:x_max-12,y_min+12:y_max-18].copy()
               # cv.imshow("patch", patch_2)
                #k+=1
                image = cv.cvtColor(patch_2,cv.COLOR_BGR2GRAY)
                image_m_blur = cv.medianBlur(image,5)
                image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
                image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
                _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv.erode(thresh, kernel)
                img_gray=cv.cvtColor(patch,cv.COLOR_BGR2GRAY)
                #cv.imshow("patch", thresh)
                print(np.mean(thresh))
                if(np.mean(thresh)>253):
                    f.write("o")
                    g.write("o")
                else:
                    f.write("x")
                    ok=0
                    while ok!=1:
                        for k in range(0,9):
                            res = cv.matchTemplate(img_gray,template[k],cv.TM_CCOEFF_NORMED)
                            loc = np.where( res >= threshold)
                            if len(loc[0]>1):
                                g.write(str(k+1))
                                ok=1
                                break
                        threshold-=0.03
                    
                cv.waitKey(0)
                cv.destroyAllWindows()
            f.write("\n")
            g.write("\n")
        #return k

img = cv.imread(r'C:\Users\Costangy\Desktop\an3sem1\analiza metanumerica\imagini\20.jpg')
img = cv.resize(img,(0,0),fx=0.2,fy=0.2)
#result=preprocess_image(img)
#img_crop=img[result[0][1]:result[2][1],result[0][0]:result[1][0]]
img_crop=preprocess_image(img)
#show_image("crop",img_crop)

#print(result)

lines_vertical=[]
for i in range(0,int(img_crop.shape[0]),int(img_crop.shape[0]/9)-1):
    l=[]
    l.append((i,0))
    l.append((i,499))
    lines_vertical.append(l)

lines_horizontal=[]
for i in range(0,math.ceil(img_crop.shape[1])+int(img_crop.shape[1]/9),math.ceil(img_crop.shape[1]/9)+1):
    l=[]
    l.append((0,i))
    l.append((499,i))
    lines_horizontal.append(l)
    
for line in  lines_vertical :
    cv.line(img_crop, line[0], line[1], (0, 255, 0), 5)
    for line in  lines_horizontal :
        cv.line(img_crop, line[0], line[1], (0, 0, 255), 5)
    #cv.imshow("img", img_crop)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
get_results(img,lines_horizontal,lines_vertical)
f.close()
g.close()