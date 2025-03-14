import numpy as np
from matplotlib.image import imread
from PIL import Image

def gaussian_filter(init_image, filter_size=(5,5), sigma=1):
    
    # Constructing the kernel
    x_filter_size = filter_size[0]
    y_filter_size = filter_size[1]

    x_filter = np.zeros((x_filter_size,1))
    y_filter = np.zeros((1,y_filter_size))

    for y in range(0, y_filter_size):
        y_filter[0][y] = 1/(np.sqrt(2*np.pi)*np.power(sigma, 2)) * np.exp(- np.power(y-y_filter_size//2, 2)/(2*np.power(sigma, 2)))

    for x in range(0, x_filter_size):
        x_filter[x][0] = 1/(np.sqrt(2*np.pi)*np.power(sigma, 2)) * np.exp(- np.power(x-x_filter_size//2, 2)/(2*np.power(sigma, 2)))

    kernel = x_filter * y_filter

    # Convolution
    y_img_s, x_img_s, c_img = init_image.shape

    h = y_filter_size//2    #1/2 the number of pixels that will be removed after the convolution(height)
    w = x_filter_size//2    #1/2 the number of pixels that will be removed after the convolution(width)

    filtered_image = np.zeros((y_img_s-2*h, x_img_s-2*w, c_img))

    for c in range(c_img):
        for i in range(h, y_img_s-h):
            for j in range(w, x_img_s-w):
                # for m in range(y_filter_size):
                #     for n in range(x_filter_size):
                #         filtered_image[i-h][j-w][c] = filtered_image[i-h][j-w][c] + kernel[y_filter_size-m-1][x_filter_size-n-1]*init_image[i-h+m][j-w+n][c]

                # Very slow for scale >=020; using np instead
                block = init_image[i-h:i-h+y_filter_size, j-w:j-w+x_filter_size, c]

                filtered_image[i-h][j-w][c] = np.sum(kernel * block)

    return np.ceil(filtered_image).astype(np.uint8)

def box_filter(init_image, filter_size=(5,5)):
    
    # Constructing the kernel
    x_filter_size = filter_size[1]
    y_filter_size = filter_size[0]

    kernel = 1/(x_filter_size*y_filter_size) * np.ones(filter_size)

    # Convolution
    y_img_s, x_img_s, c_img = init_image.shape

    h = y_filter_size//2    #1/2 the number of pixels that will be removed after the convolution(height)
    w = x_filter_size//2    #1/2 the number of pixels that will be removed after the convolution(width)

    filtered_image = np.zeros((y_img_s-2*h, x_img_s-2*w, c_img))

    for c in range(c_img):
        for i in range(h, y_img_s-h):
            for j in range(w, x_img_s-w):
                # Very slow for scale >=020; using np instead

                #for m in range(y_filter_size):
                #    for n in range(x_filter_size):
                #        filtered_image[i-h][j-w][c] = filtered_image[i-h][j-w][c] + kernel[y_filter_size-m-1][x_filter_size-n-1]*init_image[i-h+m][j-w+n][c]

                block = init_image[i-h:i-h+y_filter_size, j-w:j-w+x_filter_size, c]

                filtered_image[i-h][j-w][c] = np.sum(kernel * block)

    return filtered_image.astype(np.uint8)



image = imread('images/image_name.jpg')
kernel_size = (3,3)

filtered_image = gaussian_filter(image, kernel_size, 1)

im = Image.fromarray(filtered_image, 'RGB')
#im.save("EX2_Results/010_gauss_3.jpeg")
im.save("Demo/010_gauss_3.jpeg")


filtered_image = box_filter(image, kernel_size)

im = Image.fromarray(filtered_image, 'RGB')
#im.save("EX2_Results/010_box_3.jpeg")
im.save("Demo/010_box_3.jpeg")
