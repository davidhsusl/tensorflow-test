from FileUtils import get_file, convert_to_tf_records, read_tf_records_and_copy_images
from ModelUtils import train_model, analyze_image


def write_tf_records():
    # 資料集的位置
    train_dataset_dir = './dataset/train_orig_25'
    train_dataset_dir1 = './dataset/train'

    # 取回所有檔案路徑
    images, labels = get_file(train_dataset_dir1)
    # print('===final result===')
    # print(images)
    # print(labels)

    # 開始寫入 TFRecord 檔案
    convert_to_tf_records(images, labels, './Train.tfrecords')


def read_tf_records_and_copy():
    # TFRecord 的位置
    train_record_file = './Train.tfrecords'
    output_dir = './dataset/images'
    image_copy_count = 100
    image_pixel = 25

    # 開始讀取 TFRecord 並將影像資料寫出
    read_tf_records_and_copy_images(train_record_file, output_dir, image_copy_count, image_pixel)


# 訓練模型
def train():
    tf_filename = './Train.tfrecords'
    batch_size = 256
    label_size = 5
    save_path = './model/test_model'
    is_continue = False
    image_pixel = 25

    train_model(tf_filename, batch_size, label_size, save_path, is_continue, image_pixel)


# 辨識圖像
def image_test():
    model_path = './model/test_model-8100'
    image_path = './dataset/train_orig_25/5/pig3.jpg'
    image_pixel = 25

    analyze_image(model_path, image_path, image_pixel)


if __name__ == '__main__':
    # write_tf_records()
    # read_tf_records_and_copy()
    train()
    # image_test()
