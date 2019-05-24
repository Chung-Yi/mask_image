import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    # desk parameters
    desk_coo = [(843, 182), (0, 490), (0, 626), (265, 717), (389, 717), (1111, 206)]
    desk_color = (0, 255, 0)

    # person parameters
    person_location = [(220, 80), (390, 360)]
    person_color = (255, 0, 0, 255)

    image = cv2.imread('no_person.jpg')
    image = image[:, :, ::-1]

    desk_mask = np.zeros(image.shape, dtype=np.uint8)
    desk_coo = np.array([desk_coo], dtype=np.int32)
    cv2.fillPoly(desk_mask, desk_coo, desk_color)
    

    # dilate
    dilate_color = (255, 0, 0)
    dilate_size = (200, 200)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_size)
    dilateimg = cv2.dilate(desk_mask, dilate_kernel)
    detect_mask = cv2.bitwise_xor(desk_mask, dilateimg)

    
    mask = cv2.inRange(desk_mask, desk_color, desk_color)
    detect_mask[0 < mask] = dilate_color

    combined_image = cv2.addWeighted(image, 1, desk_mask, 0.5, 0)
    combined_image = cv2.addWeighted(combined_image, 0.5, detect_mask, 0.8, 0)

    
    person_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(person_mask, person_location[0], person_location[1], dilate_color, -1)
    # overlay area
    person_area = cv2.bitwise_and(detect_mask, person_mask)
    crop_area = person_area[person_location[0][1]:person_location[1][1], person_location[0][0]:person_location[1][0], ]
    percent = crop_area.any(axis=-1).sum() / (crop_area.shape[0] * crop_area.shape[1]) *100
    print("overlay: {:.1f}%".format(percent))

    cv2.rectangle(combined_image, person_location[0], person_location[1], person_color, 4)

    plt.imshow(combined_image)
    plt.show()





    # plt.imshow(image)
    # plt.show()
    # plt.imshow(combined_image)
    # plt.show()
    # plt.imshow(dilateimg)
    # plt.show()
    # plt.imshow(desk_mask)
    # plt.show()
    # plt.imshow(detect_mask)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()



if __name__ == '__main__':
    main()