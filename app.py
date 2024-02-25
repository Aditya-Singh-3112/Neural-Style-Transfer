import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image as process_im
import pprint
import IPython.display
import cv2
import csv

pp = pprint.PrettyPrinter(indent=4)
vgg = tf.keras.applications.VGG16(include_top=False, weights= 'imagenet')

vgg.trainable = False
pp.pprint(vgg)

def img_preprocess_driver(image):
    image_arr = np.array(image)
    image = Image.fromarray(image_arr)
    max_dim=512
    factor=max_dim/max(image.size)
    image=image.resize((round(image.size[0]*factor),round(image.size[1]*factor)))
    im_array = process_im.img_to_array(image)
    im_array = np.expand_dims(im_array,axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(im_array)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x,0)
    assert len(x.shape) == 3
    x[ :, :, 0] += 85
    x[ :, :, 1] += 90
    x[ :, :, 2] += 100
    x = np.clip(x , 0, 255).astype('uint8')
    return x

content_layers = ['block5_conv1',
                'block5_conv2']

style_layers = ['block1_conv1',
               'block2_conv1',
               'block3_conv1',
               'block4_conv1']

def get_content_loss(generated_img, content_img):
    content_loss = tf.reduce_sum(tf.square(content_img - generated_img))/(4*512*288*3)
    return content_loss

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vector = tf.reshape(tensor, [-1, channels])
    n =tf.shape(vector)[0]
    gram_matrix = tf.matmul(vector,vector, transpose_a = True)
    return gram_matrix/tf.cast(n, tf.float32)

def get_style_loss(generated_img, style_img):
    gram_noise = gram_matrix(generated_img)
    style_loss = tf.reduce_mean(tf.square(style_img - gram_noise))
    return style_loss

def compute_loss(model, loss_weights, image, gram_style_features, content_features):
    beta, alpha = loss_weights
    output = model(image)
    content_loss = 0
    style_loss = 0
    
    generated_img_style_feature = output[:len(style_layers)]
    generated_img_content_feature = output[len(style_layers):]
    
    alpha_per_layer = 1.0/float(len(content_layers))
    for a,b in zip(generated_img_content_feature, content_features):
        content_loss += alpha_per_layer*get_content_loss(a[0],b)
    
    beta_per_layer = 1.0/float(len(style_layers))
    for a,b in zip(gram_style_features, generated_img_style_feature):
        style_loss += beta_per_layer*get_style_loss(b[0], a)
    
    style_loss *= beta
    content_loss *= alpha
    
    total_loss = content_loss + style_loss
    
    return total_loss, content_loss, style_loss

def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_losses = compute_loss(**dictionary)
    total_loss = all_losses[0]
    y = tape.gradient(total_loss, dictionary['image'])
    return y, all_losses

def get_model():
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_output = content_output + style_output
    return Model(vgg.input, model_output)

def get_features_driver(model, content_img, style_img):
    content_img = img_preprocess_driver(content_img)
    style_img = img_preprocess_driver(style_img)
    
    content_output = model(content_img)
    style_output = model(style_img)
    
    content_feature = [layer[0] for layer in content_output[len(style_layers):]]
    style_feature = [layer[0] for layer in style_output[:len(style_layers)]]
    return content_feature, style_feature
    
def driver_fn(content_img, style_img, epochs, alpha =20, beta = 1e-32):
    model = get_model()
    
    for layer in model.layers:
        layer.trainable = False
    
    content_feature, style_feature = get_features_driver(model, content_img, style_img)
    
    style_gram_matrix = [gram_matrix(feature) for feature in style_feature]
    
    generated_img = img_preprocess_driver(content_img)
    generated_img = tf.Variable(generated_img, tf.float32)
    
    optimizer = Adam(learning_rate = 5, beta_1 = 0.99, epsilon = 1e-1)
    
    best_loss, best_img = 0.5, None
    
    loss_weights = (alpha, beta)
    dictionary = {'model' : model,
                 'loss_weights' : loss_weights,
                 'image' : generated_img,
                 'gram_style_features' : style_gram_matrix,
                 'content_features' : content_feature}
    
    norm_means = np.array([103.939, 116.779, 123.68])
    
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    imgs = []
   
    for i in range(0,epochs):
        grad, all_losses = compute_grads(dictionary)
        total_loss, content_loss, style_loss = all_losses
        optimizer.apply_gradients([(grad, generated_img)])
        clipped = tf.clip_by_value(generated_img, min_vals, max_vals)
        generated_img.assign(clipped)

        if total_loss<best_loss:
            best_loss = total_loss
            best_img = deprocess_img(generated_img.numpy())
            
        plot_img = generated_img.numpy()
        plot_img = deprocess_img(plot_img)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
        imgs.append(plot_img)

    return best_img, best_loss, imgs

import gradio as gr
def transfer(content_img, style_img, steps, content_weight = 20, style_weight = 1e-32):
     best_img, best_loss,images = driver_fn(content_img, style_img, epochs=steps, alpha = content_weight, beta = style_weight)
     image = Image.fromarray(images[len(images) -1])
     return image
ui = gr.Interface(transfer, inputs = [gr.Image(), gr.Image(), gr.Slider(0,200), gr.Slider(0, 50), gr.Slider(0, 1)], outputs = 'image')
ui.launch()