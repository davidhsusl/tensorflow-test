from FileUtils import get_file, convert_to_tf_record, read_tf_record
from ModelUtils import train_model, analyze_image


def main():
    # 資料集的位置
    train_dataset_dir = './dataset_train'

    # 取回所有檔案路徑
    images, labels = get_file(train_dataset_dir)
    # print('===final result===')
    # print(images)
    # print(labels)

    # 開始寫入 TFRecord 檔案
    convert_to_tf_record(images, labels, './Train.tfrecords')


def main2():
    # TFRecord 的位置
    train_record_file = './Train.tfrecords'

    # 開始讀取 TFRecord 並將影像資料寫出
    read_tf_record(train_record_file, './images')


# 訓練模型
def train():
    tf_filename = './Train.tfrecords'
    batch_size = 256
    label_size = 5
    save_path = './model/test_model'
    is_continue = True
    train_model(tf_filename, batch_size, label_size, save_path, is_continue)


# 辨識圖像
def image_test():
    model_path = './model/test_model'
    image_path = './dataset_train/5/pig3.jpg'
    analyze_image(model_path, image_path)


if __name__ == '__main__':
    # main()
    # main2()
    train()
    # image_test()
