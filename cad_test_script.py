import glob
import cv2
import numpy as np
import tensorflow as tf


class CADTest(object):
    def __init__(self):
        PATH_TO_MODEL = 'frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def processImage(cadtestobj, image):
    # heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    rows = image.shape[0]
    cols = image.shape[1]
    output_image = np.copy(image)
    # out_img, boxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colors, xstart)
    boxes, scores, classes, num = cadtestobj.get_classification(image)
    # Add heat to each box in box list
    # heat = add_heat(heat, boxes)

    # Apply threshold to help remove false positives
    # heat = apply_threshold(heat, heat_threshold)

    # Visualize the heatmap when displaying
    # heatmap = np.clip(heat, 0, 255)
    # print(boxes, scores, classes, num, sep='\n')
    # Find final boxes from heatmap using label function
    # labels = label(heatmap)

    # draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # num_detections = int(out[0][0])
    # print(num,type(num))
    for i in range(int(num[0])):

        classId = classes[0][i]
        score = scores[0][i]
        # bbox = [float(v) for v in boxes[i]]
        bbox = boxes[0][i]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            output_image = cv2.rectangle(output_image, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51),
                                         thickness=2)
    # return draw_img

    return output_image


# model = pickle.load(open('frozen_inference_graph.pb', 'rb'))
# print(model)
# exit(0)
cadtestobj = CADTest()
print("CADTest Object Created!")
images = glob.glob('data/JPEGImages/*.png')
count = 0
for ori_image in images:
    # t = time.time()
    # image = mpimg.imread(image)

    '''heatmap_img = (np.dstack((heatmap, heatmap, heatmap))*255).astype(np.uint8)
    stack1 = np.hstack((heatmap_img, out_img))
    stack2 = cv2.resize(stack1, (draw_img.shape[1], 360))
    stack3 = np.vstack((stack2, draw_img))
    plt.imshow(stack3)
    plt.figure()'''
    # plt.imshow(image)
    # plt.figure()
    # image = mpimg.imread(ori_image)
    image = cv2.imread(ori_image)
    image = cv2.resize(image, (300, 300))
    out_img = processImage(cadtestobj, image)
    # plt.imshow(out_img)
    # plt.figure()
    # plt.imshow(heatmap)
    # plt.figure()
    # plt.imshow(draw_img)
    # plt.figure()
    # t2 = time.time()
    # print(round(t2 - t, 2), 'Seconds to process image..')

    # mpimg.imsave(ori_image,out_img)
    cv2.imwrite('outputAll/' + str(count) + '.png', out_img)
    count += 1
