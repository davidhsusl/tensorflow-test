import cv2
import tensorflow as tf

from FileUtils import read_and_decode


def train_model(tf_filename, batch_size, label_size, save_path, is_continue):
    # 從 TFRecords 讀取資料並解碼
    image_batch, label_batch = read_and_decode(tf_filename, batch_size)

    # 轉換陣列的形狀
    image_batch_train = tf.reshape(image_batch, [-1, 225 * 225])

    # 把 Label 轉換成獨熱編碼
    label_batch_train = tf.one_hot(label_batch, label_size)

    # 設定訓練的對象
    w = tf.Variable(tf.zeros([225 * 225, label_size]))
    b = tf.Variable(tf.zeros([label_size]))

    # 設定影像輸入資料
    x = tf.placeholder(tf.float32, [None, 225 * 225])

    # 參數預測的結果
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    # 每張影像的正確標籤
    y_ = tf.placeholder(tf.float32, [None, label_size])

    # 計算最小交叉
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    # 使用梯度下降法來找最佳解
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    # 計算預測正確率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 計算 y 向量的最大值
    y_pred = tf.argmax(y, 1)

    # 建立 tf.train.Saver 物件
    saver = tf.train.Saver()

    # 將輸入與輸出值加入集合
    tf.add_to_collection('input', x)
    tf.add_to_collection('output', y_pred)

    # 開始訓練
    with tf.Session() as sess:
        # 初始化變數
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 建立執行緒協調器
        coord = tf.train.Coordinator()

        # 啟動文件隊列，開始讀取文件
        threads = tf.train.start_queue_runners(coord=coord)

        global count, train_accuracy
        count = 0
        train_accuracy = 0.0
        if is_continue:
            # 重建 model
            last_ckp = tf.train.latest_checkpoint('./model')
            saver = tf.train.import_meta_graph(last_ckp + '.meta')
            saver.restore(sess, last_ckp)
            count = int(last_ckp.split('-')[-1])
            print('init_value: %d' % count)

        # 訓練直到準確率到達 90%
        while train_accuracy < 0.9:
            # 讀取資料
            image_data, label_data = sess.run([image_batch_train, label_batch_train])

            # 送資料進去訓練
            sess.run(train_step, feed_dict={x: image_data, y_: label_data})

            # 每訓練 10 次後，把最新的正確率顯示出來
            if count % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
                print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy * 100))

                # 存檔
                spath = saver.save(sess, save_path, global_step=count)
                # print("Model saved in file: %s" % spath)

            count += 1

        # 關閉文件隊列
        coord.request_stop()
        coord.join(threads)

        # 把整張計算圖存檔
        # spath = saver.save(sess, save_path)
        # print("Model saved in file: %s" % spath)


def analyze_image(model_path, image_path):
    with tf.Session() as sess:
        # 使用 import_meta_graph 載入計算圖
        saver = tf.train.import_meta_graph(model_path + ".meta")

        # 使用 restore 重建計算圖
        saver.restore(sess, model_path)

        # 取出集合內的值
        x = tf.get_collection('input')[0]
        y = tf.get_collection('output')[0]

        # 讀取影像
        img = cv2.imread(image_path, 0)

        # 辨識影像，並印出結果
        result = sess.run(y, feed_dict={x: img.reshape(-1, 225 * 225)})

        print(result)
        return result
