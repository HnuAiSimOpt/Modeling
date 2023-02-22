import multiprocessing
import os
import tensorflow as tf
import random
import numpy as np
import cv2
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import sys
import math
import matplotlib.pylab as pl
import tf_geometric as tfg

def genrandint(inf, sup, num):
    y = []
    while len(y) < num:
        x = np.random.randint(inf, sup)
        count = 0
        for _, a in enumerate(y):
            if x == a:
                count += 1
        if count == 0:
            y.append(x)
    return np.sort(y)
        
def grab(x, index):
    y = []
    for _, ind in enumerate(index):
        y.append(x[ind])
    return y

def process_graph_seq(path_f, path_e, batch_size):
    Fs = os.listdir(path_f)
    Es = os.listdir(path_e)
    N = len(Fs)
    selected_index = genrandint(0, N, batch_size)
    selected_fs = grab(Fs, selected_index)
    selected_es = grab(Es, selected_index)
    graph_seq = []
    for i in range(batch_size):
        fs = tf.convert_to_tensor(np.load(path_f + selected_fs[i]), dtype=tf.float32)
        es = tf.convert_to_tensor(np.load(path_e + selected_es[i]), dtype=tf.float32)
        fs = tf.expand_dims(fs, axis=-1)
        es = tf.expand_dims(es, axis=-1)
        graph_seq.append(tf.concat([fs, es], axis=-1))
    graph_seq_batch = tf.stack(graph_seq, axis=0)
    return graph_seq_batch

def process_seq_graph(path_f, batch_size):
    Fs = os.listdir(path_f)
    N = len(Fs)
    selected_index = genrandint(0, N, batch_size)
    selected_fs = grab(Fs, selected_index)
    graph_seq = []
    for i in range(batch_size):
        fs = tf.convert_to_tensor(np.load(path_f + selected_fs[i]), dtype=tf.float32)
        graph_seq.append(fs)
    graph_seq_batch = tf.stack(graph_seq, axis=0)

    return graph_seq_batch

def process_seq(path, fv_path, latent_path, batch_size, resize, seq_length):
    folders = os.listdir(path)
    # fvs = sio.loadmat(fv_path)['fv']
    # latents = sio.loadmat(latent_path)['latent']
    n_ori = len(os.listdir(path + folders[0]))
    sc = math.floor(n_ori/seq_length)
    selected_index = genrandint(0, len(folders), batch_size)
    # selected_index = genrandint(0, 140, batch_size)
    selected_folder = grab(folders, selected_index)
    # fv_batch = tf.cast(tf.convert_to_tensor(grab(fvs, selected_index)), dtype=tf.float32)
    # latents_batch = tf.convert_to_tensor(grab(latents, selected_index))
    seq_batch = []
    for _, folder in enumerate(selected_folder):
        imgs = os.listdir(path + folder)
        seq = []
        for i in range(seq_length):
            img = tf.io.read_file(path + folder + '/' + imgs[i*sc])
            img = tf.image.decode_png(img, channels=3)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, resize)
            img = ((img / 255.0) * 2) - 1.0
            seq.append(img)
        seq = tf.stack(seq, axis=0)
        seq_batch.append(seq)
    seq_batch = tf.stack(seq_batch, axis=0)
    ################################################################
    # selected_index = genrandint(0, len(folders), batch_size)
    # latents_gan = tf.convert_to_tensor(grab(latents, selected_index))
    latents_gan = 0.
    latents_batch = 0.
    fv_batch = 0.
    return latents_gan, seq_batch, fv_batch, latents_batch

def process_matrix(matrix_path, dvs_path, latent_path, bs, use_gan=False):
    sample_list = os.listdir(matrix_path)
    n_samples = len(sample_list)
    selected_index = genrandint(0, n_samples, bs)
    selected_samples = grab(sample_list, selected_index)
    training_batch = []
    latent_batch = []
    dv_batch = []
    for sample in selected_samples:
        training_batch.append(tf.convert_to_tensor(np.load(matrix_path + sample), dtype=tf.float32))
        latent_batch.append(tf.convert_to_tensor(np.load(latent_path + sample), dtype=tf.float32))
        dv_batch.append(tf.convert_to_tensor(np.load(dvs_path + sample), dtype=tf.float32))
    training_batch = tf.stack(training_batch, axis=0)
    # training_batch = ((training_batch / 255.0)*2.0) - 1.0
    latent_batch = tf.stack(latent_batch, axis=0)
    dv_batch = tf.stack(dv_batch, axis=0)

    if use_gan:
        selected_index = genrandint(0, n_samples, bs)
        selected_samples = grab(sample_list, selected_index)
        gan_matrix_batch = []
        gan_latent_batch = []
        for sample in selected_samples:
            gan_matrix_batch.append(tf.convert_to_tensor(np.load(matrix_path + sample), dtype=tf.float32))
            gan_latent_batch.append(tf.convert_to_tensor(np.load(latent_path + sample), dtype=tf.float32))
        gan_matrix_batch = tf.stack(gan_matrix_batch, axis=0)
        gan_latent_batch = tf.stack(gan_latent_batch, axis=0)
    else:
        gan_matrix_batch = []
        gan_latent_batch = []
    return training_batch, dv_batch, latent_batch, gan_matrix_batch, gan_latent_batch


def read(path, save_path):
    files = os.listdir(path)
    fv_seq = []
    for i, file in enumerate(files):
        fo = open(path + file, "r")
        lines = fo.readlines()
        larray = lines[0].split()
        p = tf.convert_to_tensor(np.array(larray, dtype=np.float), dtype=tf.float32)
        fv_seq.append(p)
        print(i)
    fv_seq = tf.convert_to_tensor(fv_seq)
    sio.savemat(save_path, {'fv':fv_seq.numpy()})
    
def read2(path, save_path):
    files = os.listdir(path)
    fv_seq = []
    for i, file in enumerate(files):
        fo = open(path + file, "r")
        lines = fo.readlines()
        p = []
        for j, line in enumerate(lines):
            p.append(lines[j].split()[0])
        p = tf.convert_to_tensor(np.array(p, dtype=np.float), dtype=tf.float32)
        fv_seq.append(p)
        print(i)
    fv_seq = tf.convert_to_tensor(fv_seq)
    sio.savemat(save_path, {'fv':fv_seq.numpy()}) 

def trans(img_path, fv_path, save_path):
    names = os.listdir(img_path)
    fv_seq = []
    for i, name in enumerate(names):
        file = open(fv_path + name + '.txt', "r")
        lines = file.readlines()
        p = []
        for j, line in enumerate(lines):
            if j % 3 == 0:
                p.append(lines[j].split()[0])
        p = tf.convert_to_tensor(np.array(p, dtype=np.float), dtype=tf.float32)
        fv_seq.append(p)
        print(i)
    fv_seq = tf.convert_to_tensor(fv_seq)
    sio.savemat(save_path, {'fv':fv_seq.numpy()}) 
        
def op_seq_img_in_folders(path, new_path):
    folders = os.listdir(path)
    for i, folder in enumerate(folders):
        imgs = os.listdir(path + folder)
        if len(folder) == 1:
            new_folder = '00' + folder
        if len(folder) == 2:
            new_folder = '0' + folder
        if len(folder) == 3:
            new_folder = folder
        # if i <= 9:
        #     new_folder = '00' + str(i)
        # elif i <= 99:
        #     new_folder = '0' + str(i)
        # else:
        #     new_folder = str(i)
            
        os.mkdir(new_path + new_folder)
        for j, img in enumerate(imgs):
            pic = tf.io.read_file(path + folder + '/' + img)
            pic = tf.image.decode_png(pic, channels=3)
            pic = tf.cast(pic, tf.float32)
            pic = tf.image.crop_to_bounding_box(pic, offset_height=20, offset_width=425, target_height=700, target_width=700)
            pic = tf.cast(pic, tf.uint8)
            if len(img) == 5:
                new_img = '00' + img
            if len(img) == 6:
                new_img = '0' + img
            if len(img) == 7:
                new_img = img
            # new_img = img[-7:]
            # print(new_img)
            pic = pic[: , : , : : -1]
            cv2.imwrite(new_path + new_folder + '/' + new_img, pic.numpy())
            print('folder:', int(i), 'image:', int(j))



def op_txt_to_mat(path, new_path):
    files = os.listdir(path)
    p = []
    for i, doc in enumerate(files):
        fo = open(path + doc, "r")
        lines = fo.readlines()
        fv = tf.convert_to_tensor(np.array(lines[0].split(), dtype=np.float), dtype=tf.float32)
        p.append(fv)
        print(i)
    p = tf.convert_to_tensor(p)
    print(tf.shape(p))
    sio.savemat(new_path, {'fv':p.numpy()})



if __name__ == '__main__':
    training_batch, dv_batch, latent_batch = process_matrix(
    'G:/beam_crashing_dataset_80/force/normed_train/', 
    'G:/beam_crashing_dataset_80/dvs/', 
    'G:/beam_crashing_dataset_80/force/latents/', 
    10)
    print(tf.shape(training_batch))
    print(tf.shape(dv_batch))
    print(tf.shape(latent_batch))



# op_xlsx_to_mat(path="E:/Phase Field - The closed-loop optimization0102/Phase Field - The closed-loop optimization0102/data.xlsx",
        #   new_path='E:/data.mat')


# op_seq_img_in_folders(path='E:/crash/2D_view_accelerate/impact_images/', new_path='E:/crash/acc_train/')

# op_txt_to_mat(path='E:/crash/2D_view_accelerate/impact_maxValue/', new_path='E:/crash/fvs.mat')

###### from .png doc. name to .mat ######
# files = os.listdir('E:/crash/acc_test/')
# p = []
# for i, doc in enumerate(files):
#     fo = open('E:/crash/2D_view_accelerate/impact_maxValue/' + str(doc) + '.txt', "r")
#     lines = fo.readlines()
#     fv = tf.convert_to_tensor(np.array(lines[0].split(), dtype=np.float), dtype=tf.float32)
#     p.append(fv)
#     print(i)
# p = tf.convert_to_tensor(p)
# print(tf.shape(p))
# sio.savemat('E:/crash/test_fvs.mat', {'fv':p.numpy()})

###### nomalization of the feature values ######
# fvs = tf.convert_to_tensor(sio.loadmat('E:/crash/test_fvs.mat')['fv'], dtype=tf.float32)
# fvs = tf.split(fvs, 6, -1)
# normed = []
# for i, fv in enumerate(fvs):
#     if i == 0 or i == 1 or i == 2:
#         normed.append((fv - 4.0)*2.5)
#     if i == 3:
#         normed.append((fv - 30.0))
#     if i == 4:
#         normed.append((fv - 70.0)/2.0)
#     if i == 5:
#         normed.append((fv - 140.0)/2.0)
# normed = tf.transpose(tf.convert_to_tensor(normed, dtype=tf.float32))
# normed = tf.squeeze(normed)
# sio.savemat('E:/crash/test_fvs.mat', {'fv':normed.numpy()})
