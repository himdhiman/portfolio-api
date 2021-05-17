import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, GRU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def PreProcessImage(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels = 3)
  img = tf.image.resize(img, (224, 224))
  img = preprocess_input(img)
  return img, path

class Encoder(Model):
  def __init__(self, Embedding_dims):
    super(Encoder, self).__init__()
    self.fc = Dense(Embedding_dims)

  def call(self, x):
    x = self.fc(x)
    x = tf.nn.relu(x)
    return x

class LocalAttentionDecoder(Model):
  def __init__(self, units, vocab_size, Embedding_dims):
    super(LocalAttentionDecoder ,self).__init__()
    self.units = units
    self.Embedding = Embedding(vocab_size, Embedding_dims)
    self.GRU = GRU(self.units, return_sequences = True, return_state = True, recurrent_initializer = 'glorot_uniform')
    self.fc1 = Dense(self.units)
    self.Dropout = Dropout(0.5)
    self.BN = BatchNormalization()
    self.fc2 = Dense(vocab_size)
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, x, features, hidden):
    hidden_with_time = K.expand_dims(hidden, axis = 1)
    attention_score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time))
    attention_weights = tf.nn.softmax(self.V(attention_score), axis = 1)
    context_vector = attention_weights*features
    context_vector = tf.reduce_sum(context_vector, axis = 1)
    x = self.Embedding(x)
    x = K.concatenate([K.expand_dims(context_vector, axis = 1), x], axis = -1)
    output, state = self.GRU(x)
    x = self.fc1(output)
    x = K.reshape(x, (-1, x.shape[2]))
    x = self.Dropout(x)
    x = self.BN(x)
    x = self.fc2(x)
    return x, state

  def reset_state(self, batch_size):
    return K.zeros((batch_size, self.units))

def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

def train_step(img_tensor, target, decoder, encoder, tokenizer, loss_object, optimizer):
  loss = 0
  hidden = decoder.reset_state(batch_size = target.shape[0])
  dec_input = K.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
  with tf.GradientTape() as tape:
      features = encoder(img_tensor)
      for i in range(1, target.shape[1]):
          predictions, hidden = decoder(dec_input, features, hidden)
          loss += loss_function(target[:, i], predictions, loss_object)
          dec_input = K.expand_dims(target[:, i], 1)
  total_loss = (loss / int(target.shape[1]))
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss, total_loss

def evaluate(image, decoder, encoder, tokenizer, feature_model):
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(PreProcessImage(image)[0], 0)
    img_tensor_val = feature_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(39):
        predictions, hidden = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result
        dec_input = tf.expand_dims([predicted_id], 0)
    return result

def predict():
    vgg_model = VGG16(include_top = False, weights = 'imagenet')
    feature_model = Model(vgg_model.input, vgg_model.layers[-1].output)
    with open(os.path.join(BASE_DIR,"captionbot","tokenizer.pkl"), 'rb') as f:
        tokenizer = pickle.load(f)
    encoder = Encoder(256)
    decoder = LocalAttentionDecoder(512, 5001, 256)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    num_steps = 1
    img_tensor = K.zeros((64, 49, 512))
    target = K.zeros((64, 39))
    batch_loss, t_loss = train_step(img_tensor, target, decoder, encoder, tokenizer, loss_object, optimizer)
    encoder.load_weights(os.path.join(BASE_DIR,"captionbot","encoder2.hdf5"))
    decoder.load_weights(os.path.join(BASE_DIR,"captionbot","decoder2.hdf5"))
    img_name = os.listdir(os.path.join(BASE_DIR, "media", "captionbot"))[0]
    img_path = os.path.join(BASE_DIR, "media", "captionbot", img_name)
    result = evaluate(img_path, decoder, encoder, tokenizer, feature_model)
    if result[-1] == '<end>':
        caption = ""
        for i in range(len(result)-1):
            caption = caption + " " + result[i]
    else:
        caption = ""
        for i in range(len(result)):
            caption = caption + " " + result[i]
    print(caption)
    return caption
