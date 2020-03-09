# Demet Çalışkan Mef University 02.02.2020
# This program takes an input image from the user and detects the characters in it
# then compare the character with the original image characters that the program already has.
# It does that with the help of moment features. 
# After it makes the comparison it gives us the information of which character is what.
from PIL import *
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import math
from math import sqrt


def main():
    img2 = GUI()
    img = Image.open('/Users/demet/Desktop/numbers.png')
    hu1, r1, z1, numbers = images(img)
    hu2, r2, z2, numbers2 = images(img2)
    kNN(hu1, hu2, numbers)


# This method is taking an image as an input from the user
# There is no input
# Return is the image
def GUI():
    top = tk.Tk()
    top.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("png files", "*.png"), ("all files", "*.*")))
    img = Image.open(top.filename)
    return img


# This method is taking an image and finding the moments of it
# The input is image
# Returns are hu (2D array that contains hu moment), r (2D array that contains r moment), z (2D array that contains zernike moment), numbers (ordered numbers that program read)
def images(img):
    img_gray = img.convert('L')
    ONE = 150
    a = np.asarray(img_gray)
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)
    im_label, label, k = blob_coloring_8_connected(a_bin, ONE)
    rectangles = rectangles_array(im_label, k)
    new_img2 = np2PIL_color(label)
    rectangled_image = draw_rectangles(rectangles, new_img2)
    rectangled_image.show()
    hu, r, z, numbers = crop_image(im, rectangles, ONE)
    return hu, r, z, numbers

# This method converts numpy array to colored image
# Input is the im which is a numpy array
# Return is the img which is image
def np2PIL_color(im):
    img = Image.fromarray(np.uint8(im))
    return img


# This method is converts numpy array to binary array
# Inputs are im (2D numpy array), T (the comparison value), LOW (lower value that needs to be assigned), HIGH (higher value that need to be assigned)
# Return is im_out which is a binary array with 0's and 150's
def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out

# This method is eight connected blob coloring algorithm with two pass
# Inputs are bim which is binary image, ONE is a constant 150
# Outputs are im which is 2D binary array, color_im which is 3D binary array, k which is the amount of different numbers that we colored
def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    im = np.zeros(shape=(nrow,ncol), dtype = int)
    a = np.zeros(shape=max_label, dtype = int)
    a = np.arange(0,max_label, dtype = int)
    color_map = np.zeros(shape = (max_label,3), dtype= np.uint8)
    color_im = np.zeros(shape = (nrow, ncol,3), dtype= np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][1] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][2] = np.random.randint(0,255,1,dtype = np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c = bim[i][j]
                l = bim[i][j - 1]
                u = bim[i - 1][j]
                ul = bim[i - 1][j - 1]
                ur = bim[i - 1][j + 1]
                label_u = im[i - 1][j]
                label_l = im[i][j - 1]
                label_ul = im[i - 1][j - 1]
                label_ur = im[i - 1][j + 1]

                im[i][j] = max_label
                if c == ONE:
                    min_label = min(label_u, label_l, label_ul, label_ur)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label:
                            update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label:
                            update_array(a, min_label, label_l)

                        if min_label != label_ul and label_ul != max_label:
                            update_array(a, min_label, label_ul)

                        if min_label != label_ur and label_ur != max_label:
                            update_array(a, min_label, label_ur)

                else:
                    im[i][j] = max_label
    # final reduction in label array
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    #second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return im, color_im, k

# This method updates an array and changes the values inside of it
# Inputs are: a is a binary array, label1 and label2 which are values of some indexes of the array a
# It doesn't return anything since arrays can be modified
def update_array(a, label1, label2) :
    index = lab_small = lab_large = 0
    if label1 < label2 :
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else: #a[index] == lab_small
            break

    return


# This method finds the corners that has max and min coordinates for each character
# Input is label which is a 2D numpy array, k_max which is the max number of different numbers that we colored
# Output is k_table which is a 2D array that contains the max min coordinates for each character
def rectangles_array(label, k_max):
    row, column = size_of_an_array(label)
    k = 1
    min_i = row
    min_j = column
    max_i = 0
    max_j = 0
    k_table = np.zeros(shape=(k_max, 5), dtype=int)
    while k < k_max+1:
        for i in range(1, row - 1):
            for j in range(1, column - 1):
                if label[i][j] == k:
                    if i <= min_i:
                        min_i = i

                    if i >= max_i:
                        max_i = i

                    if j <= min_j:
                        min_j = j

                    if j >= max_j:
                        max_j = j

                    k_table[k][0] = k
                    k_table[k][1] = min_i
                    k_table[k][3] = max_i
                    k_table[k][2] = min_j
                    k_table[k][4] = max_j
        k = k + 1
        min_i = row
        min_j = column
        max_i = 0
        max_j = 0

    return k_table

# This method is drawing the rectangles for each character of an image
# Inputs are arr which is a numpy 2D binary array, img which is an image
# Output is the img which is an image that has rectangles on it
def draw_rectangles(arr, img):
    counter = 0
    row, column = size_of_an_array(arr)
    for i in range(0, row):
        if arr[i][0] != 0:
            counter = counter + 1
            min_i = arr[i][1]
            max_i = arr[i][3]
            min_j = arr[i][2]
            max_j = arr[i][4]
            draw = ImageDraw.Draw(img)
            draw.rectangle(((min_j, min_i), (max_j, max_i)), outline="#ff0000")
            del draw

    return img


# This method is cropping the image for each character and finding the moments for them
# Inputs are img which is an image, rectangles which is an 2D array, ONE is the constant 150
# Returns are hu_moment (2D array that contains hu moment), r_moment (2D array that contains r moment), z_moment (2D array that contains zernike moment), numbers (ordered numbers that program read)
def crop_image(img, rectangles, ONE):
    row, column = size_of_an_array(rectangles)
    hu_moment = np.zeros(shape=(10, 7), dtype=float)
    r_moment = np.zeros(shape=(10, 10), dtype=float)
    z_moment = np.zeros(shape=(10, 10), dtype=float)
    count = 0
    for i in range(0, row):
        if rectangles[i][0] != 0:
            min_i = rectangles[i][1]
            max_i = rectangles[i][3]
            min_j = rectangles[i][2]
            max_j = rectangles[i][4]
            box = (min_j, min_i, max_j, max_i)
            cropped_image = img.crop(box)
            resized_image = cropped_image.resize((21, 21))
            arr_resized = np.asarray(resized_image)
            a_bin = threshold(arr_resized, ONE/3, 0, 1)
            hu, r = hu_r_moments(a_bin, ONE)
            z = zernike_moments(a_bin, ONE)
            for i in range(7):
                hu_moment[count][i] = hu[i]
            for i in range(10):
                r_moment[count][i] = r[i]
            for i in range(10):
                z_moment[count][i] = z[i]
            count = count + 1
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    return hu_moment, r_moment, z_moment, numbers


# This method is calculating the hu and r moments of the cropped images for each character
# Inputs are arr which is an binary array of our image, ONE is the constant 150
# Outputs are H which is a 1D array that contains hu moments, R which is a 1D array that contains r moments
def hu_r_moments(arr, ONE):
    row, column = size_of_an_array(arr)
    m = np.zeros(shape=(row, column), dtype=float)
    mu = np.zeros(shape=(row, column), dtype=float)
    n = np.zeros(shape=(row, column), dtype=float)
    for k in range(1):
        for l in range(1):
            for i in range(row):
                for j in range(column):
                    m[k][l] += (i**k)*(j**l)*(arr[i][j]/ONE)
    x0 = m[1][0]/(m[0][0])
    y0 = m[0][1]/(m[0][0])
    for k in range(3):
        for l in range(3):
            for i in range(row):
                for j in range(column):
                    mu[k][l] += ((i-x0)**k)*((j-y0)**l)*(arr[i][j]/ONE)
                    y = ((k+l)/2)+1
                    n[k][l] += mu[k][l]/np.exp(mu[0][0]**y)

    H = np.zeros(7)
    H1 = n[2][0] + n[0][2]
    H2 = ((n[2][0] - n[0][2])**2) + 4*(n[1][1]**2)
    H3 = ((n[3][0] - 3*n[1][2])**2) + ((3*n[2][1] - n[0][3])**2)
    H4 = ((n[3][0] + n[1][2])**2) + ((n[2][1] + n[0][3])**2)
    H5 = (n[3][0] - 3*n[1][2])*(n[3][0] + n[1][2])*(((n[3][0] + n[1][2])**2) - 3*((n[2][1] + n[0][3])**2)) + (3*n[2][1] - n[0][3])*(n[2][1] + n[0][3])*(3*((n[3][0] + n[1][2])**2) - ((n[2][1] + n[0][3])**2))
    H6 = (n[2][0] - n[0][2])*(((n[3][0] + n[1][2])**2) - ((n[2][1] + n[0][3])**2)) + 4*n[1][1]*(n[3][0] + n[1][2])*(n[2][1] + n[0][3])
    H7 = (3*n[2][1] - n[0][3])*(n[3][0] + n[1][2])*(((n[3][0] + n[1][2])**2) - 3*((n[2][1] + n[0][3])**2)) + (3*n[1][2] - n[3][0])*(n[0][3] + n[2][1])*(3*((n[3][0] + n[1][2])**2) - ((n[2][1] + n[0][3])**2))
    H = [H1, H2, H3, H4, H5, H6, H7]

    R = np.zeros(10)
    R1 = (H2**(1/2))/H1
    R2 = (H1 + (H2**(1/2))) / (H1 - (H2**(1/2)))
    R3 = (H3**(1/2))/(H4**(1/2))
    R4 = (H3**(1/2))/(abs(H5)**(1/2))
    R5 = (H4**(1/2))/(abs(H5)**(1/2))
    R6 = abs(H6)/(H1*H3)
    R7 = abs(H6)/(H1*(abs(H5)**(1/2)))
    R8 = abs(H6)/(H3*(H2**(1/2)))
    R9 = abs(H6)/((H2*abs(H5))**(1/2))
    R10 = abs(H5)/(H3*H4)
    R = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]

    return H, R


# This method is calculating the zernike moments of the cropped images for each character
# Inputs are arr which is an binary array of our image, ONE is the constant 150
# Output is Z which is a 1D array that contains zernike moments
def zernike_moments(arr, ONE):
    row, column = size_of_an_array(arr)
    p = np.zeros(shape=(row, column), dtype=float)
    teta = np.zeros(shape=(row, column), dtype=float)
    delta_xi = 0
    delta_yj = 0
    for i in range(row-1):
        N = row
        xi = ((2**(1/2))/(N-1))*i - 1/(2**(1/2))
        delta_xi = 2/(N*(2**(1/2)))
        for j in range(column-1):
            N = column
            yj = ((2**(1/2))/(N-1))*j - 1/(2**(1/2))
            delta_yj = 2 / (N * (2 ** (1 / 2)))
            p[i][j] = (xi**2 + yj**2)**(1/2)
            if xi==0:
                teta[i][j] = np.arccos(xi)
            else:
                teta[i][j] = np.arctan(yj/xi)

    Z = np.zeros(10)
    Z11 = zernike_features(1, 1, p, teta, ONE, arr, delta_xi, delta_yj)
    Z22 = zernike_features(2, 2, p, teta, ONE, arr, delta_xi, delta_yj)
    Z31 = zernike_features(3, 1, p, teta, ONE, arr, delta_xi, delta_yj)
    Z33 = zernike_features(3, 3, p, teta, ONE, arr, delta_xi, delta_yj)
    Z42 = zernike_features(4, 2, p, teta, ONE, arr, delta_xi, delta_yj)
    Z44 = zernike_features(4, 4, p, teta, ONE, arr, delta_xi, delta_yj)
    Z51 = zernike_features(5, 1, p, teta, ONE, arr, delta_xi, delta_yj)
    Z62 = zernike_features(6, 2, p, teta, ONE, arr, delta_xi, delta_yj)
    Z64 = zernike_features(6, 4, p, teta, ONE, arr, delta_xi, delta_yj)
    Z66 = zernike_features(6, 6, p, teta, ONE, arr, delta_xi, delta_yj)
    Z = [Z11, Z22, Z31, Z33, Z42, Z44, Z51, Z62, Z64, Z66]

    return Z


# This method is calculating the zernike polinomials in order to calculate zernike features
# Inputs are n and m which are the values that zernike moments should take, p is the ro value that calculated and stored in p array
# Output R is the sum of radial zernike polinomials
def radial_zernike_polinomials(n, m, p):
    row, column = size_of_an_array(p)
    R = 0
    for i in range(row - 1):
        for j in range(column - 1):
            for s in range(int(float(n-abs(m))/2)):
                R += (((-1)**s)*(p[i][j]**(n-2*s))*(math.factorial(n-s)))/(math.factorial(abs(((n+abs(m))/2)-2))*(math.factorial(abs(((n-abs(m))/2)-2))))
    return R


# This method is calculating the zernike features in order to find zernike moments
# Inputs are n and m which are the values that zernike moments should take, p is the ro value that calculated and stored in p array,
# teta is the 2D array containing teta values,
# ONE is the constant 150, f is the array of our image, delta_xi and delta_yj are the calculated delta values
# Output is the Z which is the value of zernike feature
def zernike_features(n, m, p, teta, ONE, f, delta_xi, delta_yj):
    row, column = size_of_an_array(f)
    R = radial_zernike_polinomials(n, m, p)
    ZR = 0
    ZI = 0
    for i in range(row - 1):
        for j in range(column - 1):
            #V = R*(math.cos(m*teta[i][j])+j*math.cos(m*teta[i][j]))
            ZR += (f[i][j] / ONE) * R * (math.cos(teta[i][j])) * delta_xi * delta_yj
            ZI += (f[i][j] / ONE) * R * (math.sin(teta[i][j])) * delta_xi * delta_yj
    ZR = ZR * ((n + 1) / math.pi)
    ZI = ZI * -((n + 1) / math.pi)
    Z = ((ZR ** 2) * (ZI ** 2)) ** (1 / 2)
    return Z


# This method is finding the row and column size of an 2D array
# Input arr is the 2D array
# Outputs are row (row size of the array) and column (column size of the array)
def size_of_an_array(arr):
    row = 0
    column = 0
    for i in range(len(arr)):
        row = row + 1
        column = 0
        for j in range(len(arr[i])):
            column = column + 1
    return row, column


# This method calculate the Euclidean distance between two moments
# Input moment2 is the moment array of the second picture, moment1 is the moment array of the first picture
# Output is the distance between two moments
def euclidean_distance(moment1, moment2):
    distance = 0.0
    for i in range(len(moment2)-1):
        distance += (moment1[i] - moment2[i])**2
    return math.sqrt(distance)


# This method is putting the distances for each character to a 2d array
# Input moment2 is the moment array of the second picture, moment1 is the moment array of the first picture
# Output is the distances which is a 2D array that contains distances for each character
def get_neighbors(moment1, moment2):
    index = 0
    distances = [[]]
    for moments in moment1:
        distance = euclidean_distance(moments, moment2)
        distances.append([index, distance])
        index = index + 1
    del distances[0]
    distances.sort(key=lambda x: x[1])
    return distances


# This method is finding the most similar characters to an input image of characters
# Input moment2 is the moment array of the second picture, moment1 is the moment array of the first picture, numbers is the ordered numbers that program read
# It doesn't return anything but prints out the similar character that the program calculated
def kNN(moment1, moment2, numbers):
    min_index = 0
    f = open('/Users/demet/Desktop/kNN.txt', "w+")
    for i in range(len(moment2)):
        distances = get_neighbors(moment1, moment2[i])
        min_index = distances[i][0]
        num = numbers[min_index]
        print(min_index, "is most similar to", num)
        m = str(min_index)
        n = str(num)
        print(m + " is most similar to " + n + '\n', file=f)
    f.close()


if __name__=='__main__':
    main()
