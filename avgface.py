#!/usr/bin/python3.6

from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import random
import tensorflow as tf
import numpy as np

z_dim = 1024
img_w = 40
img_h = 40

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
  img = img.resize((img_w, img_h), Image.LANCZOS)
  return np.array(img)/255.0

# 画像のIDを学習用等に分割
image_id = list(range(1, image_num+1))
random.seed(1234)
random.shuffle(image_id)
train_id = image_id[:-2000]
valid_id = image_id[-2000:-1000]
test_id = image_id[-1000:]

# モデル
def encoder(x):
  with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
    h = tf.reshape(x, [-1, img_w*img_h*3])
    h = tf.layers.dense(h, 1024, tf.nn.relu)
    h = tf.layers.dense(h, 1024, tf.nn.relu)
    mean = tf.layers.dense(h, 1024)
    var = tf.layers.dense(h, z_dim, tf.nn.softplus)
    return mean, var

def decoder(z):
  with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    h = z
    h = tf.layers.dense(h, 1024, tf.nn.relu)
    h = tf.layers.dense(h, 1024, tf.nn.relu)
    h = tf.layers.dense(h, img_w*img_h*3, tf.nn.sigmoid)
    y = tf.reshape(h, [-1, img_w, img_h, 3])
    return y

sess = tf.Session()

# 学習
def train():
  def tf_log(x):
    return tf.log(tf.maximum(x, 1e-10))

  x = tf.placeholder(tf.float32, [None, img_w, img_h, 3])
  mean, var = encoder(x)
  z = mean + tf.sqrt(var) * tf.random_normal(tf.shape(mean))
  y = decoder(z)

  KL = -0.5*tf.reduce_mean(tf.reduce_sum(1+tf_log(var)-mean**2-var, axis=1))
  entropy = tf.reduce_mean(tf.reduce_sum(x*tf_log(y) + (1-x)*tf_log(1-y), axis=[1, 2, 3]))
  lower_bound = [-KL, entropy]
  cost = -tf.reduce_sum(lower_bound)
  optimize = tf.train.AdamOptimizer().minimize(cost)

  sess.run(tf.global_variables_initializer())

  batch_size = 1000
  for idx in range(0, 10000, batch_size):
    img = [read_image(id) for id in train_id[idx:idx+batch_size]]
    _, lb = sess.run([optimize, lower_bound], feed_dict={x: img})
    print(idx, lb)

train()

# 潜在変数の平均値を求める
x = tf.placeholder(tf.float32, [None, img_w, img_h, 3])
img = [read_image(id) for id in valid_id]
z, _ = sess.run(encoder(x), feed_dict={x: img})
z_avg = np.mean(z, axis=0)
print("z_avg", z_avg.tolist())

# 顔を平均顔に近づける
num = 16
org = [read_image(id) for id in test_id[:num]]
z_in, _ = sess.run(encoder(x), feed_dict={x: org})

z = tf.placeholder(tf.float32, [None, z_dim])

result = [[None]*11 for _ in range(num)]
for i in range(0, 11):
  p = i/10  # [0.0, 1.0]
  res = sess.run(decoder(z), feed_dict={z: z_in*(1-p)+z_avg*p})
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
