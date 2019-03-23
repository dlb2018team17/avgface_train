#!/usr/bin/python3.6

from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import random
import tensorflow as tf
import numpy as np

"""
x (image) -> [encoder] -> z (latent) -> [decoder] -> y (image)
"""

z_dim = 256
image_w = 160
image_h = 160

# 画像の入ったZIPを開く
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Align&Cropped Images
zip_name = r"img_align_celeba.zip"
zipfile = ZipFile(zip_name)
zipinfo = zipfile.infolist()
image_num = 202599
for i in range(1, image_num+1):
  assert zipinfo[i].filename=="img_align_celeba/%06d.jpg"%i

# 画像を読み出す
# idが負数ならば左右を反転
def read_image(id):
  bin = zipfile.read(zipinfo[abs(id)])
  img = Image.open(BytesIO(bin))
  # 中央160x160を切り出す
  left = (178-160)//2
  top = (218-160)//2
  img = img.crop((left, top, left+160, top+160))
  #img = img.resize((image_w, image_h), Image.LANCZOS)
  img = np.array(img)/255.0
  if id>=0:
    return img
  else:
    return img[:, ::-1, :]

# 画像のIDを学習用等に分割
image_id = list(range(1, image_num+1))
random.seed(1234)
random.shuffle(image_id)
train_id = image_id[:-1000]
test_id = image_id[-1000:]
train_id += [-id for id in train_id]
random.shuffle(train_id)

# モデル
def encoder(x, training):
  with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
    h = x
    # (160, 160, 3)
    h = tf.layers.conv2d(h, 32, (4, 4), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (80, 80, 32)
    h = tf.layers.conv2d(h, 64, (4, 4), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (40, 40, 64)
    h = tf.layers.conv2d(h, 128, (4, 4), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (20, 20, 128)
    h = tf.layers.conv2d(h, 256, (4, 4), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (10, 10, 256)
    h = tf.layers.conv2d(h, 256, (4, 4), (2, 2), padding="same")
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (5, 5, 256)
    h = tf.layers.flatten(h)
    # (6400)
    h = tf.layers.dense(h, 512)
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (512)
    mean = tf.layers.dense(h, z_dim)
    var = tf.layers.dense(h, z_dim, tf.nn.softplus)
    # (256)
  return mean, var

def decoder(z, training):
  with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    h = z
    # (256)
    h = tf.layers.dense(h, 512)
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (512)
    h = tf.layers.dense(h, 6400)
    h = tf.layers.batch_normalization(h, training=training)
    h = tf.nn.relu(h)
    # (6400)
    h = tf.reshape(h, (-1, 5, 5, 256))
    # (5, 5, 256)
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
    h = tf.layers.conv2d_transpose(h, 3, (3, 3), (2, 2), padding="same")
    h = tf.nn.sigmoid(h)
    # (160, 160, 3)
  return h

class empty:
  pass

def tf_log(x):
    return tf.log(tf.maximum(x, 1e-10))

def make_vae(x):
  mean, var = encoder(x, True)
  z = mean + tf.sqrt(var) * tf.random_normal((z_dim,))
  y = decoder(z, True)

  # 潜在変数の平均と分散の正規分布からの距離
  loss1 = -0.5*tf.reduce_mean(tf.reduce_sum(1+tf_log(var)-mean**2-var, axis=1))
  # 出力画像と元画像の二乗誤差
  loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(x-y), axis=(1, 2, 3)))

  optimizer = tf.train.AdamOptimizer().minimize(
    loss = loss1*0.01 + loss2,
    var_list = (
      tf.trainable_variables("encoder") +
      tf.trainable_variables("decoder")))

  ret = empty()
  ret.x = x
  ret.z = mean
  ret.loss1 = loss1
  ret.loss2 = loss2
  ret.optimizer = optimizer
  return ret

# 潜在変数を求める
def make_conv(x):
  z, _ = encoder(x, False)

  ret = empty()
  ret.x = x
  ret.z = z
  return ret

# 潜在変数の平均値を埋込み、画像と平均値に近づける度合いを入力して、
# 変化させた画像と潜在変数を出力する推論用モデルを作成
def make_serving(x, z_avg):
  p = tf.placeholder(tf.float32, (None,))
  z1, _ = encoder(x, False)
  p_tmp = tf.reshape(p, (-1, 1))
  z2 = z1*(1-p_tmp) + z_avg*(p_tmp)
  y = decoder(z2, False)

  ret = empty()
  ret.x = x
  ret.p = p
  ret.y = y
  ret.z = z2
  return ret

# 各関数内で定義するとエラーになるので、引数で渡す
placeholder_x = tf.placeholder(tf.float32, (None, image_w, image_h, 3))

G = make_vae(placeholder_x)
Z = make_conv(placeholder_x)

update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
#tf.train.Saver().restore(sess, "checkpoint/model.ckpt-2")

batch_size = 256
epoch_num = 16

for epoch in range(epoch_num):
#for epoch in range(3, epoch_num):
  print("epoch", epoch)

  # 学習
  z_sum = np.array([0.0]*z_dim)

  for idx in range(0, len(train_id), batch_size):
    image = [read_image(id) for id in train_id[idx:idx+batch_size]]

    _, _, G_loss1, G_loss2, z = sess.run(
      fetches = [update, G.optimizer, G.loss1, G.loss2, G.z],
      feed_dict = {G.x: image})

    z_sum += np.sum(z, axis=0)

    print(epoch, idx, G_loss1, G_loss2)

  z_avg = z_sum/len(train_id)
  print("z_avg", z_avg.tolist())

  # 最後は全画像の潜在変数をあらためて求める
  if epoch==epoch_num-1:
    z_sum = np.array([0.0]*z_dim)
    for idx in range(0, len(train_id), batch_size):
      image = [read_image(id) for id in train_id[idx:idx+batch_size]]
      z = sess.run(Z.z, feed_dict={Z.x: image})
      z_sum += np.sum(z, axis=0)
      print(epoch, idx)
    z_avg = z_sum/len(train_id)
    print("z_avg", z_avg.tolist())

  # 平均化用モデル
  S = make_serving(placeholder_x, z_avg)

  # 結果を出力
  num = 32
  image_in = []
  p = []
  for i in range(num):
    for j in range(11):
      image_in += [read_image(test_id[i])]
      p += [j/10]
  image_out = sess.run(S.y, feed_dict={S.x: image_in, S.p: p})

  canvas = Image.new("RGB", (image_w*12, image_h*num))
  for i in range(num):
    # 左端はオリジナル画像
    canvas.paste(Image.fromarray((image_in[i*11]*255).astype("uint8")),
      (0, i*image_h))
    # 徐々に平均顔に近づけた画像
    for j in range(0, 11):
      canvas.paste(Image.fromarray((image_out[i*11+j]*255).astype("uint8")),
        ((1+j)*image_w, i*image_h))
  canvas.save("report/result_%d.png"%epoch)

  # チェックポイント
  tf.train.Saver().save(sess, "checkpoint/model.ckpt", global_step=epoch)

# モデルを保存
tf.saved_model.simple_save(
  sess,
  "model",
  {"input": S.x, "p": S.p},
  {"output": S.y, "z": S.z})

print("end")
