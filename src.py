import numpy as np
from numpy import asarray 
from PIL import Image
import matplotlib.pyplot as plt
import time

def Feature():
    print("0. All the feature below")
    print("1. Brighten image")
    print("2. Adjust contrast")
    print("3. Flip image")
    print("4. RBG to BNW or Sepia")
    print("5. Blur/sharpen image")
    print("6. Crop image from center")
    print("7. Circle frame")
    print("8. 2 ellipse frame")
    return (int(input("Your choice: ")))
  
def Input(ImageName): #Read numpy array from the image
    img = Image.open(ImageName)
    numpydata = asarray(img)
    return numpydata
  
def Reshaping2d(img_3d): #Reshape from 3d to 2d matrix
    img_shape = img_3d.shape
    img_2d = img_3d.reshape(img_shape[0]*img_shape[1],img_shape[2])
    return img_2d
    
def Reshaping3d(img_2d,shape_3d): #Reshape from 2d to 3d matrix
    img_3d = img_2d.reshape((shape_3d[0],shape_3d[1],shape_3d[2]))
    img_3d = np.array(img_3d.astype(np.uint8))
    return img_3d

def Output(numpydata,FileName,Feature):
    FinalName = FileName[0] + '_' + Feature + '.png'
    data = Image.fromarray(numpydata)
    data.save(FinalName)
    plt.imshow(numpydata)
    plt.show()
    
def Limitation_Adds(img_2d,values): #Limit the highest num
    shape = img_2d.shape
    lim = 255 - values
    adds = np.full_like(img_2d,0,dtype = np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (img_2d[i][j] + values) >= lim:
                adds[i][j] = lim
            else:
                adds[i][j] = adds[i][j] + img_2d[i][j] + values
    return adds
    
def Brighten(shape_3d,img_2d,FileName):
    start = time.time()
    values = 40
    adds = Limitation_Adds(img_2d,values)
    img_3d = Reshaping3d(adds,shape_3d)
    end = time.time()
    print("Excution time_Brighten: ",end-start)
    Output(img_3d,FileName,'brighten')
   
def Contrast(shape_3d,img_2d,FileName):
    start = time.time()
    contrast_val = 50
    Minn = np.min(img_2d) + contrast_val
    Maxx = np.max(img_2d) - contrast_val
    adds = np.full_like(img_2d,0,dtype = np.uint8)
    img_2d = (np.clip(img_2d,Minn,Maxx) - Minn)/(Maxx - Minn)
    adds = (img_2d*(255-10)) + 10
    img_3d = Reshaping3d(adds,shape_3d)
    end = time.time()
    print("Excution time_Contrast: ",end-start)
    Output(img_3d,FileName,'contrast') 
    
def Flip(Choice,img_3d,FileName):
    start = time.time()
    adds = np.copy(img_3d)
    if Choice == 1:
        adds = np.flip(adds,0)
        end = time.time()
        print("Excution time_Flip Vertically: ",end-start)
        Output(adds,FileName,'flip_vertically')
    elif Choice == 2:
        adds = np.flip(adds,1)
        end = time.time()
        print("Excution time_Flip Horizontally: ",end-start)
        Output(adds,FileName,'flip_horizontally')

def rgb2gray(rgb):
    grey = rgb[0]*0.2989+rgb[1]*0.5870+rgb[2]*0.1140
    return (grey,grey,grey)

def rgb2sepia(rgb):
    tr = 0.393*rgb[0] + 0.769*rgb[1] + 0.189*rgb[2]
    if tr > 255:
        tr = 255
    tg = 0.349*rgb[0] + 0.686*rgb[1] + 0.168*rgb[2]
    if tg > 255:
        tg = 255
    tb = 0.272*rgb[0] + 0.534*rgb[1] + 0.131*rgb[2]
    if tb > 255:
        tb = 255
    return (tr,tg,tb)

def GreyscaleSepia(shape_3d,img_2d,FileName):
    start = time.time()
    #Grey scale
    shape = img_2d.shape
    grey = np.full_like(img_2d,0,dtype = np.uint8)
    for i in range(shape[0]):
        grey[i] = rgb2gray(img_2d[i])
    img_3d_gc = Reshaping3d(grey,shape_3d)
    #Sepia
    sepia = np.full_like(img_2d,0,dtype = np.uint8)
    for i in range(shape[0]):
        sepia[i] = rgb2sepia(img_2d[i])
    img_3d_sp = Reshaping3d(sepia,shape_3d)
    end = time.time()
    print("Excution time_Grayscale Sepia: ",end-start)
    Output(img_3d_gc,FileName,'greyscale')
    Output(img_3d_sp,FileName,'sepia')

def crop(image_array, crop_ratio):
    image_height, image_width, _ = image_array.shape
    crop_height = int(image_height * crop_ratio)
    crop_width = int(image_width * crop_ratio)

    top = (image_height - crop_height) // 2
    left = (image_width - crop_width) // 2
    bottom = top + crop_height
    right = left + crop_width
    cropped_image_array = image_array[top:bottom, left:right, :]
    return cropped_image_array

def resize(image_array, target_size):
    # target_size: Tuple (width, height)
    width, height = target_size
    resized_image_array = np.zeros((height, width, image_array.shape[2]), dtype=image_array.dtype)
    for y in range(height):
        for x in range(width):
            original_x = int(x / width * image_array.shape[1])
            original_y = int(y / height * image_array.shape[0])
            resized_image_array[y, x] = image_array[original_y, original_x]
    return resized_image_array

def CropImage(image_array, FileName):
    start = time.time()
    crop_ratio = 0.5 # crop_ratio: Tỉ lệ cắt ảnh (0.0 - 1.0)
    #crop image from center
    cropped_image_array = crop(image_array, crop_ratio)
    # resize to the original size
    target_size = (image_array.shape[1], image_array.shape[0])
    resized_image_array = resize(cropped_image_array, target_size)
    end = time.time()
    print("Excution time_Crop Image: ",end-start)
    Output(resized_image_array,FileName,'crop_'+str(crop_ratio*100)+'%')
    
def BoxBlur(image, FileName):
    start = time.time()
    kernel_size = 3 #odd number to define center in the kernel
    # Define the box kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    num_channels = image.shape[2]
    blurred_image = np.zeros_like(image)
    for channel in range(num_channels):
        padding = kernel_size // 2
        padded_channel = np.pad(image[:, :, channel], ((padding, padding), (padding, padding)), mode="constant")
        for i in range(padding, image.shape[0] + padding):
            for j in range(padding, image.shape[1] + padding):
                window = padded_channel[i - padding : i + padding + 1, j - padding : j + padding + 1]
                blurred_image[i - padding, j - padding, channel] = int(np.sum(window * kernel))
    blurred_image = blurred_image.astype(np.uint8)
    end = time.time()
    print("Excution time_Box Blur: ",end-start)
    Output(blurred_image,FileName,'boxblur_'+str(kernel_size)+'kernelsize')

def Sharpen(image, FileName):
    start = time.time()
    factor = 1.0
    # Define the sharpening kernel
    kernel = np.array([[0, -factor, 0],
                       [-factor, 1 + 4*factor, -factor],
                       [0, -factor, 0]])

    num_channels = image.shape[2]

    sharpened_image = np.zeros_like(image)
    for channel in range(num_channels):
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                window = image[i - 1 : i + 2, j - 1 : j + 2, channel]
                sharpened_value = np.sum(window * kernel)
                sharpened_image[i, j, channel] = np.clip(sharpened_value, 0, 255)
    end = time.time()
    print("Excution time_Sharpen: ",end-start)
    Output(sharpened_image,FileName,'sharpen_'+str(factor)+'factor')

def CircleFrame(image_array, Filename):
    start = time.time()
    color=(0, 0, 0)
    (width, height) = (image_array.shape[0],image_array.shape[1])
    radius = width//2
    center_x = width // 2
    center_y = height // 2
    # Tạo mặt nạ hình tròn với các giá trị 1 (True) trong phạm vi bán kính và các giá trị 0 (False) ngoài phạm vi bán kính
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    # Chuyển đổi hình ảnh sang numpy array
    # Áp dụng mặt nạ lên hình ảnh gốc
    masked_image = np.copy(image_array)
    masked_image[~mask] = color
    end = time.time()
    print("Excution time_Circle Frame: ",end-start)
    Output(masked_image, FileName, 'circleframe')
    
def EllipseFrame(image_array,FileName,v=0.25):
    start = time.time()
    height, width = image_array.shape[:2]
    center_x, center_y = width // 2, height // 2
    # semi_minor_axis = 150
    # semi_major_axis = np.sqrt(width ** 2 + height ** 2) / 2 - semi_minor_axis/5
    v=int(height*v)
    semi_minor_axis = np.sqrt(height*v//2)
    semi_major_axis = np.sqrt(height*(height-v)//2)
    
    y, x = np.ogrid[:height, :width]
    x_rotated = (x - center_x) * np.cos(np.pi / 4) - (y - center_y) * np.sin(np.pi / 4)
    y_rotated = (x - center_x) * np.sin(np.pi / 4) + (y - center_y) * np.cos(np.pi / 4)

    mask1 = (x_rotated / semi_major_axis) ** 2 + (y_rotated / semi_minor_axis) ** 2 <= 1
    mask2 = (x_rotated / semi_minor_axis) ** 2 + (y_rotated / semi_major_axis) ** 2 <= 1

    mask = mask1 | mask2
    masked_image = np.copy(image_array)
    masked_image[~mask] = (0,0,0)
    end = time.time()
    print("Excution time_Ellipse Frame: ",end-start)
    Output(masked_image, FileName, 'ellipseframe')
    
if __name__ == "__main__":
    ImageName = input("Please enter the input name of a file: ")
    img_3d = Input(ImageName)
    shape_3d = img_3d.shape
    img_2d = Reshaping2d(img_3d)
    FileName = ImageName.split('.')
    Choice = Feature()
    if Choice == 0:
        Brighten(shape_3d,img_2d,FileName)
        Contrast(shape_3d,img_2d,FileName)
        Flip(1,img_3d,FileName)
        Flip(2,img_3d,FileName)
        GreyscaleSepia(shape_3d,img_2d,FileName)
        BoxBlur(img_3d,FileName)
        Sharpen(img_3d,FileName)
        CropImage(img_3d,FileName)
        CircleFrame(img_3d,FileName)
        EllipseFrame(img_3d,FileName)
    if Choice == 1:
        Brighten(shape_3d,img_2d,FileName)
    if Choice == 2:
        Contrast(shape_3d,img_2d,FileName)
    if Choice == 3:
        Choice = int(input("1. Flip vertically \n2. Flip horizontally\n Your choice: "))
        Flip(Choice,img_3d,FileName)
    if Choice == 4:
        GreyscaleSepia(shape_3d,img_2d,FileName)
    if Choice == 5:
        Choice = int(input("1. Box blur \n2. Sharpen\n Your choice: "))
        if (Choice == 1): BoxBlur(img_3d,FileName)
        else: Sharpen(img_3d,FileName)
    if Choice == 6:
        CropImage(img_3d,FileName)
    if Choice == 7:
        CircleFrame(img_3d,FileName)
    if Choice == 8:
        EllipseFrame(img_3d,FileName)
    
    