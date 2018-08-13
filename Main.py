from FileUtils import get_file, convert_to_tf_records, read_tf_records_and_write
from ModelUtils import train_model, analyze_image


def write_tf_records():
    # 資料集的位置
    train_dataset_dir = './dataset/train'

    # 取回所有檔案路徑
    images, labels = get_file(train_dataset_dir)
    # print('===final result===')
    # print(images)
    # print(labels)

    # 開始寫入 TFRecord 檔案
    convert_to_tf_records(images, labels, './dataset/Train.tfrecords')


def read_tf_records():
    # TFRecord 的位置
    train_record_file = './dataset/Train.tfrecords'
    output_dir = './images'
    image_copy_count = 100

    # 開始讀取 TFRecord 並將影像資料寫出
    read_tf_records_and_write(train_record_file, output_dir, image_copy_count)


# 訓練模型
def train():
    tf_filename = './dataset/Train.tfrecords'
    batch_size = 256
    label_size = 5
    save_path = './model/test_model'
    is_continue = True
    train_model(tf_filename, batch_size, label_size, save_path, is_continue)


# 辨識圖像
def image_test():
    model_path = './model/test_model-8100'
    image_path = './dataset/train_orig/5/pig3.jpg'
    analyze_image(model_path, image_path)


if __name__ == '__main__':
    # write_tf_records()
    # read_tf_records()
    train()
    # image_test()
