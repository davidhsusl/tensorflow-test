import os
import random as r

import cv2
import numpy as np
import tensorflow as tf


# 讀取出指定資料夾下所有的檔案，在同一資料夾下的檔案給定同一 tag
def get_file(file_dir):
    # The images in each subfolder
    images = []
    # The subfolders
    subfolders = []

    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(file_dir):
        names = []
        for name in fileNames:
            names.append(os.path.join(dirPath, name))

        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))

        # print('===os.walk begin===')
        # print(names)
        # print(subfolders)

        # 隨機打亂各個資料夾內的數據
        r.shuffle(names)
        if names:
            images.append(names)

    # 計算最小檔案數量的資料夾
    min_count = float('Inf')
    for folder in subfolders:
        n_img = len(os.listdir(folder))

        if n_img < min_count:
            min_count = n_img

    # print('===images===')
    # print(images)

    # 只保留最小檔案數量
    for i in range(len(images)):
        # print('===只保留最小檔案數量 begin===')
        # print(images[i])
        images[i] = images[i][0:min_count]
        # print(images[i])

    # 二維轉一維
    # print('reshape begin')
    # print(images)
    images = np.reshape(images, [min_count * len(subfolders), ])
    # print(images)

    # To record the labels of the image dataset
    labels = []
    for count in range(len(subfolders)):
        # print('===prepare labels===')
        labels = np.append(labels, min_count * [count])
        # print(labels)

    # 打亂最後輸出的順序，去除每個類別間的隔閡
    # print('===打亂最後輸出的順序，去除每個類別間的隔閡===')
    subfolders = np.array([images, labels])
    # print(subfolders)
    subfolders = subfolders[:, np.random.permutation(subfolders.shape[1])].T
    # print(subfolders)

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


# 轉Int64資料為 tf.train.Feature 格式
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 轉Bytes資料為 tf.train.Feature 格式
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 將檔案寫成 tfRecord 格式
def convert_to_tf_records(images, labels, filename):
    global image_raw
    n_samples = len(labels)
    tf_writer = tf.python_io.TFRecordWriter(filename)

    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i], 0)

            if image is None:
                print('Error image: ' + images[i])
            else:
                image_raw = image.tostring()

            label = int(labels[i])

            # 將 tf.train.Feature 合併成 tf.train.Features
            ftrs = tf.train.Features(feature={'label': int64_feature(label), 'image_raw': bytes_feature(image_raw)})

            # 將 tf.train.Features 轉成 tf.train.Example
            example = tf.train.Example(features=ftrs)

            # 將 tf.train.Example 寫成 tfRecord 格式
            tf_writer.write(example.SerializeToString())

        except IOError as e:
            print('Error! skip!')

    tf_writer.close()
    print('Transform done!')


# 讀取 tfRecord 檔案
def read_tf_records_and_write(tf_filename, output_dir, image_copy_count):
    # 產生文件名隊列
    filename_quene = tf.train.string_input_producer([tf_filename], shuffle=False, num_epochs=image_copy_count)

    # 數據讀取器
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_quene)

    # 數據解析
    image_features = tf.parse_single_example(serialized_example,
                                             features={'label': tf.FixedLenFeature([], tf.int64),
                                                       'image_raw': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(image_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [225, 225])

    label = tf.cast(image_features['label'], tf.int64)

    with tf.Session() as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 建立執行緒協調器
        coord = tf.train.Coordinator()

        # 啟動文件隊列，開始讀取文件
        threads = tf.train.start_queue_runners(coord=coord)

        count = 0
        try:
            while count < 10000:
                # 讀取
                image_data, label_data = sess.run([image, label])

                # 寫檔
                cv2.imwrite(output_dir + '/tf_%d_%d.jpg' % (label_data, count), image_data)
                count += 1

            print('Done!')
        except tf.errors.OutOfRangeError:
            print('No more image!')
        finally:
            coord.request_stop()

        coord.join(threads)


def read_tf_recodes_by_batch(tf_filename, batch_size):
    # 產生文件名隊列
    filename_quene = tf.train.string_input_producer([tf_filename], num_epochs=None)

    # 數據讀取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_quene)

    # 數據解析
    image_features = tf.parse_single_example(serialized_example,
                                             features={'label': tf.FixedLenFeature([], tf.int64),
                                                       'image_raw': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(image_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [225, 225])

    label = tf.cast(image_features['label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=10000 + 3 * batch_size,
        min_after_dequeue=1000)

    return image_batch, label_batch
