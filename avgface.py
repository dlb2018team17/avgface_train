#!/usr/bin/python3.6

from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import random
import tensorflow as tf
import numpy as np

z_dim = 256
img_w = 160
img_h = 160

# 画像の入ったZIPを開く
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Align&Cropped Images
zip_name = r"img_align_celeba.zip"
#zipfile = ZipFile(zip_name)
# なぜか途中からディスクアクセスがボトルネックになるので最初に全体を読みこむ
zipfile = ZipFile(BytesIO(open(zip_name, "rb").read()))
print("load zip")
image_num = 202599

def read_image(id):
  bin = zipfile.read("img_align_celeba/%06d.jpg"%id)
  img = Image.open(BytesIO(bin))
  # 中央160x160を切り出す
  left = (178-160)//2
  top = (218-160)//2
  img = img.crop((left, top, left+160, top+160))
  #img = img.resize((img_w, img_h), Image.LANCZOS)
  return np.array(img)/255.0

# 画像のIDを学習用等に分割
image_id = list(range(1, image_num+1))
random.seed(1234)
random.shuffle(image_id)
train_id = image_id[:-2000]
valid_id = image_id[-2000:-1000]
test_id = image_id[-1000:]

# モデル
def encoder(x, training):
  with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
    h = x
    # (160, 160, 3)
    h = tf.layers.conv2d(h, 32, (5, 5), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (80, 80, 32)
    h = tf.layers.conv2d(h, 64, (5, 5), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (40, 40, 64)
    h = tf.layers.conv2d(h, 128, (5, 5), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (20, 20, 128)
    h = tf.layers.conv2d(h, 256, (5, 5), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (10, 10, 256)
    h = tf.layers.conv2d(h, 512, (5, 5), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (5, 5, 512)
    h = tf.layers.flatten(h)
    # (12800)
    h = tf.layers.dense(h, 1024)
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (1024)
    mean = tf.layers.dense(h, z_dim)
    var = tf.layers.dense(h, z_dim, tf.nn.softplus)
  return mean, var

def decoder(z, training):
  with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    h = z
    # (1024)
    h = tf.layers.dense(h, 12800)
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (12800)
    h = tf.reshape(h, [-1, 5, 5, 512])
    # (5, 5, 512)
    h = tf.layers.conv2d_transpose(h, 256, (3, 3), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (10, 10, 256)
    h = tf.layers.conv2d_transpose(h, 128, (3, 3), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (20, 20, 128)
    h = tf.layers.conv2d_transpose(h, 64, (3, 3), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (40, 40, 64)
    h = tf.layers.conv2d_transpose(h, 32, (3, 3), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (80, 80, 32)
    h = tf.layers.conv2d_transpose(h, 3, (3, 3), (2, 2), padding="same", activation=tf.nn.sigmoid)
    # (160, 160, 3)
  return h

sess = tf.Session()

# 学習
def train():
  def tf_log(x):
    return tf.log(tf.maximum(x, 1e-10))

  x = tf.placeholder(tf.float32, [None, img_w, img_h, 3])
  mean, var = encoder(x, True)
  z = mean + tf.sqrt(var) * tf.random_normal(tf.shape(mean))
  y = decoder(z, True)

  KL = -0.5*tf.reduce_mean(tf.reduce_sum(1+tf_log(var)-mean**2-var, axis=1))
  diff = tf.reduce_mean(tf.reduce_sum(tf.square(x-y), axis=[1, 2, 3]))
  lower_bound = [-KL*0.01, -diff]
  cost = -tf.reduce_sum(lower_bound)
  optimize = tf.train.AdamOptimizer().minimize(cost)
  update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  sess.run(tf.global_variables_initializer())

  batch_size = 256
  for epoch in range(4):
    for idx in range(0, len(train_id), batch_size):
      img = [read_image(id) for id in train_id[idx:idx+batch_size]]
      _, _, lb = sess.run([optimize, update, lower_bound], feed_dict={x: img})
      print(epoch, idx, lb)

train()

# 潜在変数の平均値を求める
x = tf.placeholder(tf.float32, [None, img_w, img_h, 3])
img = [read_image(id) for id in valid_id[:256]]
z, _ = sess.run(encoder(x, False), feed_dict={x: img})
z_avg = np.mean(z, axis=0)
print("z_avg", z_avg.tolist())

# 顔を平均顔に近づける
num = 32
org = [read_image(id) for id in test_id[:num]]
z_in, _ = sess.run(encoder(x, False), feed_dict={x: org})

z = tf.placeholder(tf.float32, [None, z_dim])

result = [[None]*11 for _ in range(num)]
for i in range(0, 11):
  p = i/10  # [0.0, 1.0]
  res = sess.run(decoder(z, False), feed_dict={z: z_in*(1-p)+z_avg*p})
  for j in range(num):
    result[j][i] = res[j]

# 結果を画像にまとめて出力
img = Image.new("RGB", (img_w*12, img_h*num))
for i in range(num):
  # 左端はオリジナル画像
  img.paste(Image.fromarray((org[i]*255).astype('uint8')), (0, i*img_h))
  # 徐々に平均顔に近づけた画像
  for j in range(0, 11):
    img.paste(Image.fromarray((result[i][j]*255).astype('uint8')), ((1+j)*img_w, i*img_h))
img.save("result.png")
