import argparse
import cv2
from matplotlib import pyplot as plt
from random import randint
import numpy as np
from keras.preprocessing.image import img_to_array



def speckle(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)

    image = (image * (randint(1, 10))) + (image * (gauss) / (randint(1, 3)))
    return image


def poisson(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals * 2))
    image = np.random.poisson((image * image * vals * ((randint(3, 9) / 10.0))) / float(vals))
    return image


def imgshow(image):
    plt.imshow(image)
    plt.show()


def gaussian(image):
    gaussian_noise = image.copy()
    cv2.randn(gaussian_noise, 0, 150)
    image = image + gaussian_noise
    return image


def scaling(image):
    row, col, ch = image.shape

    check = randint(1, 3)
    scale = randint(1, 9)
    scale2 = randint(1, 9)
    if check == 1:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (row, col))
        return image

    elif check == 2:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (row, col))
        return image

    elif check == 3:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (row, col))
        return image

    else:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (row, col))
        return image


def rotation(image):
    rows, cols, ch = image.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), randint(0, 360), 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image


def translation(image):
    rows, cols, ch = image.shape

    m = np.float32([[1, 0, int(randint(1, rows) * .45)], [0, 1, int(randint(1, cols) * .45)]])
    image = cv2.warpAffine(image, m, (cols, rows))
    return image


def affine(image):
    rows, cols, ch = image.shape

    pts1 = np.float32([[50, randint(2, 15)], [200, randint(10, 50)], [50, randint(200, 250)]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M, (cols, rows))
    return image


##Edge Detection###

def edgedetection(image):
    edges = cv2.Canny(image, 100, 200)
    img = cv2.imread(edges)
    return img


def color(image):
    image = (255 - image)
    return image


def inverse(image):
    image = np.int16(image)

    contrast = randint(1, 60)
    brightness = randint(1, 60)

    img = image * (contrast / 127 + 1) - contrast + brightness

    img = np.clip(img, 0, 255)
    image = np.uint8(img)
    return image


def img2data(image, data, i, imgsku):
    image = cv2.resize(image, (256, 256))

    for x in range(i):
        data.append(img_to_array(color(image)))
        data.append(img_to_array(image))
        data.append(img_to_array(affine(image)))
        data.append(img_to_array(translation(image)))
        data.append(img_to_array(rotation(image)))
        data.append(img_to_array(scaling(image)))
        data.append(img_to_array(gaussian(image)))
        data.append(img_to_array(poisson(image)))
        data.append(img_to_array(speckle(image)))
        data.append(img_to_array(inverse(image)))
    #
    # print(len(data))
    input_file = open('/home/manbhi1/Desktop/Datasets/active_images.csv', 'r')
    for i in range(0, 1):
        line = input_file.readline().rstrip()
    for line in input_file:
        line = line.rstrip()
        v = line.split(",")
        product_id = v[0]
        product_id = product_id.replace('"', '')
        if (product_id == imgsku):
            break

    data = np.array(data, dtype="float") / 255.0

    data = data.reshape((len(data), np.prod(data.shape[1:])))

    for i in range(len(data)):
        for j in range((len(v))):
            if j != 0 or j != 1:
                data[i] = np.append(data[i], (v[j]))

    data = np.reshape(data, (len(data), 256, 256, 3))

    return data


def random_change(image):
    x = randint(1, 10)
    if x == 1:
        return color(image)
    if x == 2:
        return image
    if x == 3:
        return affine(image)
    if x == 4:
        return translation(image)
    if x == 5:
        return rotation(image)
    if x == 6:
        return scaling(image)
    if x == 7:
        return gaussian(image)
    if x == 8:
        return poisson(image)
    if x == 9:
        return speckle(image)
    if x == 10:
        return inverse(image)


#


def perturbtion_display(image):
    image = cv2.resize(image, (256, 256))
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        img = random_change(image)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def imgs2csv(FilePaths):
    data = []
    for file in FilePaths:
        image = cv2.imread(file)
        image = cv2.resize(image, (256, 256))
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float") / 255.0

    # data = data.reshape((len(data), np.prod(data.shape[1:])))

    data = np.reshape(data, (len(data), 256, 256, 3))
    return data


def main():
   # perturbtion_display(cv2.imread(r'/home/manbhi1/Desktop/logo-blue-mobile.jpg'))



#  print("Part 1: Edge Detection and Conversion to GrayScale\n")
#
#  edgedetection(source=args.s, destination=args.d)
#
#  print("Part 2: Converting to CSV\n")
#
# # img2csv(output=args.o, destination=args.d)


if __name__ == '__main__':
    main()
