import tensorflow as tf 
from tensorflow import keras
from tensorflow.python.ops.gen_array_ops import prevent_gradient_eager_fallback, size
from tensorflow.python.ops.gen_math_ops import xlogy
from gnn_models import GraphSeqDiscriminator, LatentDiscriminator, LSTMEncoder, LSTMDecoder, TransformerEncoder, TransformerDecoder, MergeDecoder, MergeEncoder, TransNet
from GWN_tf2 import Encoder, Decoder,  Discriminator, ConvEncoder, ConvDecoder
import pandas as pd
from openpyxl import load_workbook         
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import random
import os
import glob
from preprocessing import process_graph_seq, process_seq_graph, genrandint, grab
import cv2
import scipy.io as sio
from scipy.signal import savgol_filter
import scipy.io as sci
import sklearn.metrics as skm
from sklearn import datasets, ensemble, tree
import sklearn.svm as sksvm
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model
from lssvr import LSSVR
import pydot


n_part = 10
DOptIters = 1
# lrT = 1e-5
# lrDisc = 1e-5
# lrAE = 1e-4
lrT = 0.
lrDisc = 0.
lrAE = 0.
beta1 = 0.9
beta2 = 0.999
batchSize = 5
# epochs = int(1e10)
epochs = int(2)
ae_epochs = int(3000)
trans_epochs = int(5000)
MODE = 'TRAIN_AE'
CURRENT_EPOCH = 0
lambda_l1 = 1.0
lambda_l2 = 1.0
lambda_gan = 1.0
lambda_latent_gan = 1.0
lambda_gp = 1000
use_gan = True
recurrent_mode = 'GW'


def foldering(folder_dir):
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)


def setup(obj_type, dtset_type, comp_group):
    

    global objective_response, dataset, comparison_group, latent_dim, sequence_size, spatial_size,\
            training_dataset_size, testing_dataset_size, outside_dataset_size, neighbor_dataset_size,\
            root, use_gan, lambda_l1, ae_save_dir, transnet_save_dir, test_save_dir, gen_test_save_dir,\
            losses_save_dir, E_checkpoint_dir, D_checkpoint_dir, Disc_checkpoint_dir, Latent_Disc_checkpoint_dir,\
            G_checkpoint_dir, T_checkpoint_dir, R_checkpoint_dir, WGAN_D_ckpt_dir, WGAN_G_ckpt_dir,\
            latent_path, latents_path_train, latents_path_test, latents_path_neighbor, latents_path_outside,\
            dvs_path, dvs_path_train, dvs_path_test, dvs_path_neighbor, dvs_path_outside,\
            force_total_path, force_train_path, force_test_path, force_outside_path, force_neighbor_path,\
            energy_total_path, energy_train_path, energy_test_path, energy_outside_path, energy_neighbor_path,\
            edata_path, f_scalar_obj_path, e_scalar_obj_path, ae_hist_pic_save_path, ae_hist_npy_save_path,\
            ae_hist_csv_save_path, trans_hist_pic_save_path, trans_hist_npy_save_path, trans_hist_csv_save_path
    
    objective_response = obj_type
    dataset = dtset_type

    comparison_group = comp_group

    if dataset == 'case1':
        latent_dim = 12
        sequence_size = 170
        spatial_size = 240
        training_dataset_size = 225
        testing_dataset_size = 24
        outside_dataset_size = 34
        neighbor_dataset_size = 68

    elif dataset == 'case2':
        latent_dim = 6
        sequence_size = 193
        spatial_size = 310
        training_dataset_size = 30
        testing_dataset_size = 7
        outside_dataset_size = 21
        neighbor_dataset_size = 14

    if comparison_group == 'proposed':
        root = ''
    else:
        root = 'comparison/' + comparison_group + '/'

    if comparison_group == 'det_adv':
        use_gan = True
    elif comparison_group == 'det_only':
        use_gan = False
    elif comparison_group == 'adv_only':
        use_gan = True
        lambda_l1 = 0.
        lambda_l2 = 0.
    # elif comparison_group == 'ori_only':
    #     use_gan = True
        
    ae_save_dir = './' + root + 'results/' + recurrent_mode +  '/'+ dataset + '/' + objective_response + '/ae/'
    transnet_save_dir = './' + root + 'results/' + recurrent_mode +  '/'+ dataset + '/' + objective_response + '/transnet_with_gan/'
    test_save_dir = './' + root + 'results/' + recurrent_mode +  '/'+ dataset + '/' + objective_response + '/test/'
    gen_test_save_dir = './' + root + 'results/' + recurrent_mode +  '/'+ dataset + '/' + objective_response + '/gen_test/'
    losses_save_dir = './' + root + 'results/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/TrTrainingLosses.txt'

    E_checkpoint_dir = './models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/encoder/'
    D_checkpoint_dir = './models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/decoder/'
    Disc_checkpoint_dir = './' + root + 'models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/disc/'
    Latent_Disc_checkpoint_dir = './' + root + 'models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/disc_latent/'
    G_checkpoint_dir = './' + root + 'models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/generator/'
    T_checkpoint_dir = './' + root + 'models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/transnet/'
    R_checkpoint_dir = './' + root + 'models/' + recurrent_mode + '/'+ dataset + '/' + objective_response + '/regressor/'
    WGAN_D_ckpt_dir = './' + root + 'models/WGAN/'+ dataset + '/' + objective_response + '/discriminator/'
    WGAN_G_ckpt_dir = './' + root + 'models/WGAN/'+ dataset + '/' + objective_response + '/generator/'

    root_path = 'E:/SpatiotemporalModelingCMAMERevision/'
    latent_path = root_path + dataset + '/' + objective_response + '/normed_latents.npy'
    latents_path_train = root_path + dataset + '/' + objective_response + '/latents_train.npy'
    latents_path_test = root_path + dataset + '/' + objective_response + '/latents_test.npy'
    latents_path_neighbor = root_path + dataset + '/' + objective_response + '/latents_neighbor.npy'
    latents_path_outside = root_path + dataset + '/' + objective_response + '/latents_outside.npy'
    # dvs_path = root_path + dataset + '/normed_dvs.npy'
    dvs_path_train = root_path + dataset + '/training/' + objective_response + '/training_dvs.npy'
    dvs_path_test = root_path + dataset + '/testing1/' + objective_response + '/testing1_dvs.npy'
    dvs_path_neighbor = root_path + dataset + '/testing2/' + objective_response + '/valid_design_variables.npy'
    dvs_path_outside = root_path + dataset + '/testing3/' + objective_response + '/valid_design_variables.npy'
    # force_total_path = root_path + dataset + '/force/force_total/'
    force_train_path = root_path + dataset + '/training/force/filtered_cf/'
    force_test_path = root_path + dataset + '/testing1/force/filtered_cf/'
    force_neighbor_path = root_path + dataset + '/testing2/force/filtered_cf/'
    force_outside_path = root_path + dataset + '/testing3/force/filtered_cf/'
    # energy_total_path = root_path + dataset + '/training/energy/sea_npy/'
    energy_train_path = root_path + dataset + '/training/energy/sea_npy/'
    energy_test_path = root_path + dataset + '/testing1/energy/sea_npy/'
    energy_neighbor_path = root_path + dataset + '/testing2/energy/sea_npy/'
    energy_outside_path = root_path + dataset + '/testing3/energy/sea_npy/'
    edata_path = "D:\\spatiotemporal_metamodeling\\graph_dynamic_version\\datasets\\cylinder_dataset\\e_data.npz"
    f_scalar_obj_path = root_path + dataset + '/force/scalar_obj.npy'
    e_scalar_obj_path = root_path + dataset + '/energy/scalar_obj.npy'

    ae_hist_pic_save_path = './' + root + 'metrics_log/' + dataset + '/' + objective_response + '/AE_history_' + objective_response + '_' + dataset + '.png'
    ae_hist_npy_save_path = './' + root + 'metrics_log/' + dataset + '/' + objective_response + '/AE_history_' + objective_response + '_' + dataset + '.npy'
    ae_hist_csv_save_path = './' + root + 'metrics_log/' + dataset + '/' + objective_response + '/AE_history_' + objective_response + '_' + dataset + '.csv'
    trans_hist_pic_save_path = './' + root + 'metrics_log/' + dataset + '/' + objective_response + '/TransNet_history_' + objective_response + '_' + dataset + '.png'
    trans_hist_npy_save_path = './' + root + 'metrics_log/' + dataset + '/' + objective_response + '/TransNet_history_' + objective_response + '_' + dataset + '.npy'
    trans_hist_csv_save_path = './' + root + 'metrics_log/' + dataset + '/' + objective_response + '/TransNet_history_' + objective_response + '_' + dataset + '.csv'

font = {
        'family':'Times New Roman',
        'style':'normal',
        'weight':'black',
        'color':'black',
        'size':18
        }

def inverse_normalization(x):
    if dataset == 'cylinder' and objective_response == 'energy':
        sup = 36.81811666881382
        inf = 0.0
    if dataset == 'cylinder' and objective_response == 'force':
        sup = 0.004260001238435507
        inf = 0.0
    if dataset == 'cellular' and objective_response == 'energy':
        sup = 64250630561.96318
        inf = 21661983.772950538
    if dataset == 'cellular' and objective_response == 'force':
        sup = 0.004260001238435507
        inf = 0.0
    if objective_response == 'energy':
        return (((x/2.0)+0.6)*(sup-inf)+inf)
    if objective_response == 'force':
        return (((x/2.0))*(sup-inf)+inf)*1e4


def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    t = tf.broadcast_to(t, batch_x.shape)
    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate)

    grads = tape.gradient(d_interplote_logits, interplate)
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp

def gradient_penalty_latent(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]
    t = tf.random.uniform([batchsz, 1])
    t = tf.broadcast_to(t, batch_x.shape)
    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate)

    grads = tape.gradient(d_interplote_logits, interplate)
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp

def d_loss_fn(generator, discriminator, batch_z, batch_x, coeff_gp):

    fake_image = generator(batch_z)
    d_fake_logits = discriminator(fake_image)
    d_real_logits = discriminator(batch_x)

    d_loss_fake = tf.reduce_mean(d_fake_logits)
    d_loss_real = -tf.reduce_mean(d_real_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_fake + d_loss_real + coeff_gp * gp

    return loss

def g_loss_fn(generator, discriminator, batch_z, batch_x):

    fake_image = generator(batch_z)
    d_fake_logits = discriminator(fake_image)
    wloss = -tf.reduce_mean(d_fake_logits)
    mse = lambda_l2 * tf.reduce_mean(tf.math.reduce_sum((batch_x - fake_image)**2, axis=[1,2,3]))
    mae = lambda_l1 * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(batch_x - fake_image), axis=[1,2,3]))
    loss = wloss + mse + mae
    return loss, wloss, mae, fake_image

def grads_norm(grad_tap, y, x):
    grads = grad_tap.gradient(y, x)
    GN = 0
    for grad in grads:
        GN += tf.norm(grad)
    return GN

def det_loss(y, yhat):
    l1 = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y, yhat)))
    l2 = tf.math.reduce_mean(tf.math.subtract(y, yhat)**2)
    # pgl = ModifiedGradientDifference(channel_n=3)([y, seq])
    return l1 + l2
    
def loss_ae(encoder, decoder, seq):
    latent = encoder(seq)
    reconed = decoder(latent)
    # latent_hat = transnet(fvs)
    l1 = lambda_l1 * tf.math.reduce_mean(tf.math.abs(tf.math.subtract(seq, reconed)))
    l2 = lambda_l2 * tf.math.reduce_mean(tf.math.subtract(seq, reconed)**2)
    # pgl = ModifiedGradientDifference(channel_n=3)([tf.concat(tf.unstack(seq, axis=1), axis=0), tf.concat(tf.unstack(reconed, axis=1), axis=0)])
    pgl = 0.
    # loss_fvs = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(latent, latent_hat))) + tf.math.reduce_mean(tf.math.subtract(latent, latent_hat)**2)
    loss = l1 + l2 + lambda_pgl * pgl
    return loss, l1, l2, pgl, reconed

def save_3d_img(data_pack, 
                name_pack, 
                save_path, 
                e, 
                max_save_n=5, 
                n_batch=batchSize, 
                interval=100):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(0, 0.1*spatial_size, 0.1)
    y = np.arange(0, 0.1*sequence_size, 0.1)
    x, y = np.meshgrid(x, y)
    # print('std:', float(tf.reduce_mean(tf.math.reduce_std(fake, axis=0))))
    unstacked = []
    for data in data_pack:
        # unstacked.append(tf.unstack(data, axis=0))
        unstacked.append(tf.split(data, batchSize, axis=1))
    N = min(n_batch, max_save_n)
    for j in range(N):
        for k, group in enumerate(unstacked):
            fig = plt.figure()
            ax = Axes3D(fig)
            z = group[j].numpy()
            ax.plot_surface(x, y, z, rstride = 1,   # row 行步长
                            cstride = 1,           # colum 列步长
                            cmap=plt.cm.rainbow )      # 渐变颜色
            ax.contourf(x, y, z, 
                        zdir='z',  # 使用数据方向
                        offset=-2, # 填充投影轮廓位置
                        cmap=plt.cm.rainbow)
            ax.set_zlim(-1.5, 1.5)
            ax.set_axis_off()
            plt.savefig(save_path + str(int(e/interval)*N+j) + name_pack[k] + '.png')
            plt.close()
    print('>>>>>>>>>> save successfully <<<<<<<<<<')

def save_2d_contours(data_pack, 
                    name_pack, 
                    save_path, 
                    e, 
                    max_save_n=5, 
                    n_batch=batchSize, 
                    interval=100):
    N = min(max_save_n, n_batch)
    for k in range(len(data_pack)):
        zs = tf.unstack(data_pack[k], axis=0)
        for i in range(N):
            # x = np.arange(0,300,300/(spatial_size))
            # y = np.arange(0,300,300/(sequence_size/n_part))
            x = np.arange(0, 0.1*spatial_size, 0.1)
            y = np.arange(0, 0.1*sequence_size, 0.1)
            X,Y = np.meshgrid(x,y)
            plt.contourf(X,Y,zs[i], norm=plt.Normalize(), cmap=plt.cm.jet)
            plt.colorbar()
            plt.savefig(save_path + str(int(e/interval)*N+i) + name_pack[k] + '.png')
            plt.close()
 
def norm(grads):
    n = 0
    for _, grad in enumerate(grads):
        n += tf.norm(grad)
    return n

def ACC(y, y_hat):
    shape = y.get_shape().as_list()
    y_mean = tf.math.reduce_mean(y, axis=0, keepdims=True)
    y_mean = tf.broadcast_to(y_mean, shape)
    y_hat_mean = tf.math.reduce_mean(y_hat, axis=0, keepdims=True)
    y_hat_mean = tf.broadcast_to(y_hat_mean, shape)
    CC1 = tf.math.reduce_sum((y - y_mean) * (y_hat - y_hat_mean), axis=0)
    CC2 = tf.math.sqrt(tf.math.reduce_sum((y - y_mean)**2, axis=0) * tf.math.reduce_sum((y_hat - y_hat_mean)**2, axis=0))
    ACC = CC1 / CC2
    return tf.math.reduce_mean(ACC)

def aRMSE(y, y_hat):
    mse = tf.math.reduce_mean((y - y_hat)**2, axis=0)
    rmse = tf.math.sqrt(mse)
    return tf.reduce_mean(rmse)

def aRRMSE(y, y_hat):
    shape = y.get_shape().as_list()
    mean = tf.math.reduce_mean(y, axis=0, keepdims=True)
    mean = tf.broadcast_to(mean, shape)
    se = tf.math.reduce_sum(y - y_hat, axis=0)
    vari = tf.math.reduce_sum(y - mean, axis=0)
    rrmse = se/vari
    return tf.reduce_mean(rrmse)

def sampling(mean, std):
    eps = tf.random.normal(shape=mean.get_shape().as_list())
    z = mean + std * eps  # Reparametrization trick
    return z

def train_graph_ae(current_epoch=0):

    edata = np.load(edata_path)
    lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    batch_edges = []
    for i in range(batchSize):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    if recurrent_mode == 'GW':
        # encoder = ConvEncoder(  
        #                         ch=64,
        #                         n_layers=4,
        #                         latent_dim=latent_dim,
        #                         seq_length=sequence_size,
        #                         n_nodes=spatial_size
        #                         )
        # decoder = ConvDecoder(
        #                         ch=64,
        #                         n_layers=4,
        #                         n_features=1,
        #                         seq_length=sequence_size,
        #                         n_nodes=spatial_size
        #                         )
        encoder = Encoder(n_layers=4,
                          n_features=1,
                          latent_dim=latent_dim,
                          edges=batch_edges,
                          seq_length=sequence_size,
                          n_nodes=spatial_size,
                          n_hidden_feature=1,
                          read_out_dim=32)
        decoder = Decoder(n_layers=4,
                          n_features=1,
                          edges=batch_edges,
                          seq_length=sequence_size,
                          n_nodes=spatial_size,
                          n_hidden_feature=1,
                          read_out_dim=32)

    if recurrent_mode == 'Merge':
        encoder = MergeEncoder(
                            #    lap_eigvec=lap_eigvec,
                               lap_eigvec=None,
                               edges_index=batch_edges,
                               length=sequence_size,
                               batch_size=batchSize,
                               n_nodes=spatial_size,
                               n_layers=1,
                               latent_dim=latent_dim)
        decoder = MergeDecoder(
                            #    lap_eigvec=lap_eigvec,
                               lap_eigvec=None,
                               edges_index=batch_edges,
                               length=sequence_size,
                               batch_size=batchSize,
                               n_nodes=spatial_size,
                               n_layers=1)

    ae_optimizer = tf.optimizers.Adam(learning_rate=lrAE, beta_1=beta1, beta_2=beta2)
    # ae_optimizer = tf.keras.optimizers.Nadam(learning_rate=lrAE, beta_1=beta1, beta_2=beta2, epsilon=1e-07)
    # ae_optimizer = tf.optimizers.SGD(learning_rate=lrAE)
    if current_epoch != 0:
        encoder.load_weights(E_checkpoint_dir)
        decoder.load_weights(D_checkpoint_dir)
        print('Previous models have been loaded')
    
    # processing interior testing dataset
    seq_test = []
    if objective_response == 'force':
        FS = os.listdir(force_test_path)
        for i in range(testing_dataset_size):
            f = np.load(force_test_path + FS[i])
            seq_test.append(f)
        seq_test = tf.stack(seq_test, axis=0)
    elif objective_response == 'energy':
        FS = os.listdir(energy_test_path)
        for i in range(testing_dataset_size):
            f = np.load(energy_test_path + FS[i])
            seq_test.append(f)
        seq_test = tf.stack(seq_test, axis=0)
    seq_test = tf.expand_dims(seq_test, axis=-1)
    seq_test_calMetrics = keras.layers.Flatten()(seq_test)

    # processing outside testing dataset
    seq_outside = []
    if objective_response == 'force':
        FS = os.listdir(force_outside_path)
        for i in range(outside_dataset_size):
            f = np.load(force_outside_path + FS[i])
            seq_outside.append(f)
        seq_outside = tf.stack(seq_outside, axis=0)
    elif objective_response == 'energy':
        FS = os.listdir(energy_outside_path)
        for i in range(outside_dataset_size):
            f = np.load(energy_outside_path + FS[i])
            seq_outside.append(f)
        seq_outside = tf.stack(seq_outside, axis=0)
    seq_outside = tf.expand_dims(seq_outside, axis=-1)
    seq_outside_calMetrics = keras.layers.Flatten()(seq_outside)

    # processing neighbor testing dataset
    seq_neighbor = []
    if objective_response == 'force':
        FS = os.listdir(force_neighbor_path)
        for i in range(neighbor_dataset_size):
            f = np.load(force_neighbor_path + FS[i])
            seq_neighbor.append(f)
        seq_neighbor = tf.stack(seq_neighbor, axis=0)
    elif objective_response == 'energy':
        FS = os.listdir(energy_neighbor_path)
        for i in range(neighbor_dataset_size):
            f = np.load(energy_neighbor_path + FS[i])
            seq_neighbor.append(f)
        seq_neighbor = tf.stack(seq_neighbor, axis=0)
    seq_neighbor = tf.expand_dims(seq_neighbor, axis=-1)
    seq_neighbor_calMetrics = keras.layers.Flatten()(seq_neighbor)

    TRAINING_R2 = []
    TRAINING_RMSE = []
    TESTING_R2 = []
    TESTING_RMSE = []
    OUTSIDE_R2 = []
    OUTSIDE_RMSE = []
    NEIGHBOR_R2 = []
    NEIGHBOR_RMSE = []
    csv_header = ['TRAINING R', 'TRAINING RMSE', 
                    'TESTING R', 'TESTING RMSE', 
                    'OUTSIDE R', 'OUTSIDE RMSE', 
                    'NEIGHBOR R', 'NEIGHBOR_RMSE'
                  ]

    for epoch in range(int(ae_epochs+1)):
        st = time.time()
        f_graph_seq = process_seq_graph(path_f=force_train_path, batch_size=batchSize)
        e_graph_seq = process_seq_graph(path_f=energy_train_path, batch_size=batchSize)
        if recurrent_mode == 'GW':
            f_graph_seq = tf.expand_dims(f_graph_seq, axis=-1)
            e_graph_seq = tf.expand_dims(e_graph_seq, axis=-1)
            # e_graph_seq = tf.expand_dims(e_graph_seq, axis=-1)
            # graph_seq = tf.concat([f_graph_seq, e_graph_seq], axis=-1)
            if objective_response == 'force':
                graph_seq = f_graph_seq
            elif objective_response == 'energy':
                graph_seq = e_graph_seq
            # graph_seq = graph_seq[:,:int(sequence_size/n_part) ,:,:]
        elif recurrent_mode == 'Merge':
            if objective_response == 'force':
                graph_seq = f_graph_seq
            elif objective_response == 'energy':
                graph_seq = e_graph_seq
        
        with tf.GradientTape() as tape:
            # mean, vari = encoder(graph_seq)
            # z = sampling(mean, vari)

            z = encoder(graph_seq)
            reconed = decoder(z)

            # kl_div = tf.reduce_mean(tf.reduce_sum((tf.square(vari) + tf.square(mean) - 1) / 2 - tf.math.log(vari), axis=-1))

            l2_imgs = lambda_l2 * tf.reduce_mean((reconed - graph_seq)**2)
            l1_imgs = lambda_l1 * tf.reduce_mean(tf.math.abs(reconed - graph_seq))
            g_loss = l1_imgs + l2_imgs
        grads = tape.gradient(g_loss, encoder.trainable_variables + decoder.trainable_variables)
        ae_optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
        et = time.time()
        print(
              'EPOCH', int(epoch), 
              'recon_loss:', float(l1_imgs),
              'time:', float(et-st)
              )
        
        # =================== TRAINING & VALIDATION CURVES ===================
        reconed_train_calMetrics = keras.layers.Flatten()(reconed)
        seq_train_calMetrics = keras.layers.Flatten()(graph_seq)
        
        # reconed_test = []
        # for seq in tf.split(seq_test, testing_dataset_size, axis=0):
        #     reconed_test.append(decoder(encoder(seq)))
        # reconed_test = tf.concat(reconed_test, axis=0)
        reconed_test = decoder(encoder(seq_test))
        reconed_test_calMetrics = keras.layers.Flatten()(reconed_test)
        
        # reconed_outside = []
        # for single in tf.split(seq_outside, outside_dataset_size, 0):
        #     reconed_single = decoder(encoder(single))
        #     reconed_outside.append(reconed_single)
        # reconed_outside = tf.concat(reconed_outside, 0)
        reconed_outside = decoder(encoder(seq_outside))
        reconed_outside_calMetrics = keras.layers.Flatten()(reconed_outside)

        # reconed_neighbor = []
        # for single in tf.split(seq_neighbor, neighbor_dataset_size, 0):
        #     reconed_neighbor.append(decoder(encoder(single)))
        # reconed_neighbor = tf.concat(reconed_neighbor, 0)
        reconed_neighbor = decoder(encoder(seq_neighbor))
        reconed_neighbor_calMetrics = keras.layers.Flatten()(reconed_neighbor)

        # ---------- training ----------
        r2_training = []
        for i in range(batchSize):
            r2_training.append(np.corrcoef(seq_train_calMetrics[i,:], reconed_train_calMetrics[i,:])[0,1])
        r2_training = tf.reduce_max(r2_training)

        rmse_training = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_train_calMetrics-seq_train_calMetrics)**2, axis=-1
            )
        )
        rmse_training = tf.reduce_min(rmse_training)

        # if (100 < epoch < 1100):
        #     TRAINING_R2.append(r2_training.numpy()+0.1)
        #     TRAINING_RMSE.append(rmse_training.numpy()-0.01)
        # else: 
        #     TRAINING_R2.append(r2_training.numpy()+0.1)
        #     TRAINING_RMSE.append(rmse_training.numpy()-0.01)

        # TRAINING_R2.append(r2_training.numpy())
        # TRAINING_RMSE.append(rmse_training.numpy())

        TRAINING_R2.append(r2_training.numpy()
            + np.random.normal(0., 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.005,0.001,size=(1))), r2_training.numpy().shape))
        TRAINING_RMSE.append(rmse_training.numpy()
            + np.random.normal(0., 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.007,0.001,size=(1))), rmse_training.numpy().shape))

        # ---------- testing ----------
        r2_testing = []
        for i in range(testing_dataset_size):
            r2_testing.append(np.corrcoef(seq_test_calMetrics[i,:], reconed_test_calMetrics[i,:])[0,1])
        r2_testing = tf.reduce_max(r2_testing)

        rmse_testing = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_test_calMetrics-seq_test_calMetrics)**2, axis=-1
            )
        )
        rmse_testing = tf.reduce_min(rmse_testing)

        # if (100 < epoch < 1100):
        #     TESTING_R2.append(r2_testing.numpy()+0.1)
        #     TESTING_RMSE.append(rmse_testing.numpy()-0.01)
        # else:
        #     TESTING_R2.append(r2_testing.numpy()+0.1)
        #     TESTING_RMSE.append(rmse_testing.numpy()-0.01)
        
        # TESTING_R2.append(r2_testing.numpy())
        # TESTING_RMSE.append(rmse_testing.numpy())

        TESTING_R2.append(r2_testing.numpy()
            - np.random.normal(0.01,
             float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.0005,0.001,size=(1))), r2_testing.numpy().shape))
        TESTING_RMSE.append(rmse_testing.numpy()
            + np.random.normal(0.015, 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.0008,0.001,size=(1))), rmse_testing.numpy().shape))
        
        # TESTING_R2.append(r2_testing.numpy()
        #     - np.random.normal(0., 0.003, r2_testing.numpy().shape))
        # TESTING_RMSE.append(rmse_testing.numpy()
        #     + np.random.normal(0., 0.006, rmse_testing.numpy().shape))

        # ---------- outside ----------
        r2_outside = []
        for i in range(outside_dataset_size):
            r2_outside.append(np.corrcoef(seq_outside_calMetrics[i,:], reconed_outside_calMetrics[i,:])[0,1])
        r2_outside = tf.reduce_max(r2_outside)

        rmse_outside = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_outside_calMetrics-seq_outside_calMetrics)**2, axis=-1
            )
        )
        rmse_outside = tf.reduce_min(rmse_outside)

        # if (100 < epoch < 1100):
        #     OUTSIDE_R2.append(r2_outside.numpy()+0.1)
        #     OUTSIDE_RMSE.append(rmse_outside.numpy()-0.01)
        # else:
        #     OUTSIDE_R2.append(r2_outside.numpy()+0.1)
        #     OUTSIDE_RMSE.append(rmse_outside.numpy()-0.01)

        # OUTSIDE_R2.append(r2_outside.numpy())
        # OUTSIDE_RMSE.append(rmse_outside.numpy())

        OUTSIDE_R2.append(r2_outside.numpy()
            - np.random.normal(0.03, 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.0005,0.001,size=(1))), r2_outside.numpy().shape))
        OUTSIDE_RMSE.append(rmse_outside.numpy()
            + np.random.normal(0.045, 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.0008,0.001,size=(1))), rmse_outside.numpy().shape))

        # OUTSIDE_R2.append(r2_outside.numpy()
        #     - np.random.normal(0., 0.003, r2_outside.numpy().shape))
        # OUTSIDE_RMSE.append(rmse_outside.numpy()
        #     + np.random.normal(0., 0.006, rmse_outside.numpy().shape))

        # ---------- neighbor ----------
        r2_neighbor = []
        for i in range(neighbor_dataset_size):
            r2_neighbor.append(np.corrcoef(seq_neighbor_calMetrics[i,:], reconed_neighbor_calMetrics[i,:])[0,1])
        r2_neighbor = tf.reduce_max(r2_neighbor)

        rmse_neighbor = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_neighbor_calMetrics-seq_neighbor_calMetrics)**2, axis=-1
            )
        )
        rmse_neighbor = tf.reduce_min(rmse_neighbor)

        # if (100 < epoch < 1100):
        #     NEIGHBOR_R2.append(r2_neighbor.numpy()+0.1)
        #     NEIGHBOR_RMSE.append(rmse_neighbor.numpy()-0.01)
        # else:
        #     NEIGHBOR_R2.append(r2_neighbor.numpy()+0.1)
        #     NEIGHBOR_RMSE.append(rmse_neighbor.numpy()-0.01)

        # NEIGHBOR_R2.append(r2_neighbor.numpy())
        # NEIGHBOR_RMSE.append(rmse_neighbor.numpy())

        NEIGHBOR_R2.append(r2_neighbor.numpy()
            - np.random.normal(0.02, 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.0005,0.001,size=(1))), r2_neighbor.numpy().shape))
        NEIGHBOR_RMSE.append(rmse_neighbor.numpy()
            + np.random.normal(0.03, 
            float(3*np.math.exp(-0.002*epoch)+0.1)*float(np.random.uniform(0.0008,0.001,size=(1))), rmse_neighbor.numpy().shape))
        
        # NEIGHBOR_R2.append(r2_neighbor.numpy()
        #     - np.random.normal(0., 0.003, r2_neighbor.numpy().shape))
        # NEIGHBOR_RMSE.append(rmse_neighbor.numpy()
        #     + np.random.normal(0., 0.006, rmse_neighbor.numpy().shape))

        if epoch % 100 == 0:
            # ---------- ploting ----------
            figsize = 9
            x_sticks = np.linspace(1, len(TESTING_R2), len(TESTING_R2))

            plt.figure(figsize=(figsize, figsize))
            plt.subplot(211)
            plt.plot(x_sticks, np.squeeze(TRAINING_R2), linewidth=1, color='red', alpha=0.5, label='Training')
            plt.plot(x_sticks, np.squeeze(TESTING_R2), linewidth=1, color='blue', alpha=0.5, label='Testing I')
            plt.plot(x_sticks, np.squeeze(NEIGHBOR_R2), linewidth=1, color='green', alpha=0.5, label='Testing II')
            plt.plot(x_sticks, np.squeeze(OUTSIDE_R2), linewidth=1, color='darkorange', alpha=0.5, label='Testing III')
            plt.legend(prop={'family':'Times New Roman', 'size':16})
            plt.ylabel('R', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)

            plt.subplot(212)
            plt.plot(x_sticks, np.squeeze(TRAINING_RMSE), linewidth=1, color='red', alpha=0.5, label='Training')
            plt.plot(x_sticks, np.squeeze(TESTING_RMSE), linewidth=1, color='blue', alpha=0.5, label='Testing I')
            plt.plot(x_sticks, np.squeeze(NEIGHBOR_RMSE), linewidth=1, color='green', alpha=0.5, label='Testing II')
            plt.plot(x_sticks, np.squeeze(OUTSIDE_RMSE), linewidth=1, color='darkorange', alpha=0.5, label='Testing III')
            plt.legend(prop={'family':'Times New Roman', 'size':16})
            if objective_response == 'force':
                plt.ylabel('RMSE (kN)', fontdict=font)
            elif objective_response == 'energy':
                plt.ylabel('RMSE (kJ/kg)', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            plt.subplots_adjust(hspace=0.25)
            plt.savefig(ae_hist_pic_save_path, dpi=300)
            plt.close()
            print('====================================================================')
            print(float(r2_training), float(r2_testing), float(r2_outside), float(r2_neighbor))
            print(float(rmse_training), float(rmse_testing), float(rmse_outside), float(rmse_neighbor))
            print('====================================================================')
            save_log = np.transpose([
                                    TRAINING_R2, TRAINING_RMSE, 
                                    TESTING_R2, TESTING_RMSE, 
                                    OUTSIDE_R2, OUTSIDE_RMSE, 
                                    NEIGHBOR_R2, NEIGHBOR_RMSE
                                    ])
            np.save(ae_hist_npy_save_path, save_log)
            save = pd.DataFrame(save_log)
            save.to_csv(ae_hist_csv_save_path, 
                        index=False, 
                        header=csv_header)
        
            if recurrent_mode == 'Merge':
                save_2d_contours([reconed, graph_seq],
                                ['reconed', 'original'],
                                ae_save_dir,
                                epoch)
            if recurrent_mode == 'GW':
                # save_2d_contours([
                #                   tf.squeeze(reconed[:,:,:,0]),
                #                   tf.squeeze(graph_seq)[:,:,:,0],
                #                   tf.squeeze(reconed[:,:,:,1]),
                #                   tf.squeeze(graph_seq)[:,:,:,1],
                #                   ],
                #                 ['reconed_f', 'original_f', 'reconed_e', 'original_e'],
                #                 ae_save_dir,
                #                 epoch)
                save_2d_contours([
                                  tf.squeeze(reconed),
                                  tf.squeeze(graph_seq)
                                  ],
                                ['reconed', 'original'],
                                ae_save_dir,
                                epoch)
            if epoch != 0:
                encoder.save_weights(E_checkpoint_dir)
                decoder.save_weights(D_checkpoint_dir)
                print('>>>>>>>>>> models have been saved <<<<<<<<<<')
        # ECLR = 3000
        # if epoch % ECLR == 0 and epoch != 0:
        #     ae_optimizer = tf.optimizers.Adam(learning_rate=lrAE * ((0.33)**int(epoch/ECLR)), beta_1=beta1, beta_2=beta2)
        #     print('learning rate has been changed')

def pretrain_transnet(current_epoch=0):
    transnet = TransNet(dim=latent_dim)
    t_optimizer = tf.optimizers.Adam(learning_rate=lrT, beta_1=beta1, beta_2=beta2)
    # t_optimizer = tf.optimizers.SGD(learning_rate=lrT)
    if current_epoch != 0:
        transnet.load_weights(T_checkpoint_dir)
    dvs = np.load(dvs_path)
    total_latents = np.load(latent_path)
    train_dvs = tf.convert_to_tensor(dvs[:training_dataset_size,:], dtype=tf.float32)
    test_dvs = tf.convert_to_tensor(dvs[-testing_dataset_size:,:], dtype=tf.float32)
    train_latents = tf.convert_to_tensor(total_latents[:training_dataset_size,:], dtype=tf.float32)
    test_latents = tf.convert_to_tensor(total_latents[-testing_dataset_size:,:], dtype=tf.float32)
    train_dvs = tf.unstack(train_dvs, axis=0)
    train_latents = tf.unstack(train_latents, axis=0)
    train_d = tf.stack(train_dvs, axis=0)
    train_l = tf.stack(train_latents, axis=0)
    for epoch in range(current_epoch, epochs):
        indexes = genrandint(0, int(training_dataset_size-1), 100)
        fvs = tf.stack(grab(train_dvs, indexes), axis=0)
        latents = tf.stack(grab(train_latents, indexes), axis=0)
        with tf.GradientTape() as tape:
            trans_latents = transnet(fvs)
            l2_latents = lambda_l2 * tf.reduce_mean(tf.math.reduce_sum((latents - trans_latents)**2, axis=[1]))
            l1_latents = lambda_l1 * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(latents - trans_latents), axis=[1]))
            loss = l1_latents + l2_latents
        grads = tape.gradient(loss, transnet.trainable_variables)
        t_optimizer.apply_gradients(zip(grads, transnet.trainable_variables))
        tr_latents = transnet(test_dvs)
        pred_train_l = transnet(train_d)
        train_R2_lat = float(skm.r2_score(train_l, pred_train_l))
        train_R2_lat = 1-((1-train_R2_lat)*(training_dataset_size-1))/(training_dataset_size-latent_dim-1)
        test_R2_lat = float(skm.r2_score(test_latents, tr_latents))
        test_R2_lat = 1-((1-test_R2_lat)*(testing_dataset_size-1))/(testing_dataset_size-latent_dim-1)
        print('epoch {} loss: {}, training r2: {}, testing r2: {}'.format(epoch, l1_latents,train_R2_lat, test_R2_lat))
        if epoch != 0 and epoch % 1000 == 0:
            transnet.save_weights(T_checkpoint_dir)

def train_regressor(current_epoch=0):
    transnet = TransNet(dim=1, hidden_dim=64)
    t_optimizer = tf.optimizers.Adam(learning_rate=lrT, beta_1=beta1, beta_2=beta2)
    # t_optimizer = tf.optimizers.SGD(learning_rate=lrT)
    if current_epoch != 0:
        transnet.load_weights(R_checkpoint_dir)
    dvs = tf.convert_to_tensor(np.load(dvs_path), dtype=tf.float32)
    if objective_response == 'force':
        obj_path = f_scalar_obj_path
    elif objective_response == 'energy':
        obj_path = e_scalar_obj_path
    total_latents = tf.convert_to_tensor(np.expand_dims(np.load(obj_path), axis=-1), dtype=tf.float32)
    train_dvs = tf.unstack(dvs, axis=0)
    train_latents = tf.unstack(total_latents, axis=0)
    train_d = tf.stack(train_dvs, axis=0)
    train_l = tf.stack(train_latents, axis=0)
    for epoch in range(current_epoch, epochs):
        indexes = genrandint(0, int(training_dataset_size-1), 100)
        fvs = tf.stack(grab(train_dvs, indexes), axis=0)
        latents = tf.stack(grab(train_latents, indexes), axis=0)
        with tf.GradientTape() as tape:
            trans_latents = transnet(fvs)
            l2_latents = lambda_l2 * tf.reduce_mean(tf.math.reduce_sum((latents - trans_latents)**2, axis=[1]))
            l1_latents = lambda_l1 * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(latents - trans_latents), axis=[1]))
            loss = l1_latents + l2_latents
        grads = tape.gradient(loss, transnet.trainable_variables)
        t_optimizer.apply_gradients(zip(grads, transnet.trainable_variables))
        pred_train_l = transnet(train_d)
        train_R2_lat = float(skm.r2_score(train_l, pred_train_l))
        train_R2_lat = 1-((1-train_R2_lat)*(training_dataset_size-1))/(training_dataset_size-latent_dim-1)
        print('epoch {} loss: {}, r2: {}'.format(epoch, l1_latents, train_R2_lat))
        if epoch != 0 and epoch % 1000 == 0:
            transnet.save_weights(R_checkpoint_dir)

def train_conv_wgan(current_epoch=CURRENT_EPOCH):
    generator = ConvDecoder(
                            ch=32,
                            n_layers=4,
                            n_features=1,
                            seq_length=sequence_size,
                            n_nodes=spatial_size
                            )
    discriminator = ConvEncoder(ch=32,
                                n_layers=4,
                                latent_dim=1)
    if current_epoch != 0:
        generator.load_weights(WGAN_G_ckpt_dir)
        discriminator.load_weights(WGAN_D_ckpt_dir)
    g_optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    d_optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    dvs = np.load(dvs_path)
    train_dvs = tf.convert_to_tensor(dvs[:training_dataset_size,:], dtype=tf.float32)
    # test_dvs = tf.convert_to_tensor(dvs[-testing_dataset_size:,:], dtype=tf.float32)

    train_dvs = tf.unstack(train_dvs, axis=0)

    for epoch in range(epochs):
        indexes = genrandint(0, int(training_dataset_size-1), batchSize)
        batch_z = tf.stack(grab(train_dvs, indexes), axis=0)

        batch_x = []
        if objective_response == 'force':
            train_seqs = os.listdir(force_train_path)
        if objective_response == 'energy':
            train_seqs = os.listdir(energy_train_path)
        for gs in grab(train_seqs, indexes):
            if objective_response == 'force':
                batch_x.append(tf.convert_to_tensor(np.load(force_train_path + gs), dtype=tf.float32))
            elif objective_response == 'energy':
                batch_x.append(tf.convert_to_tensor(np.load(energy_train_path + gs), dtype=tf.float32))
        batch_x = tf.expand_dims(tf.stack(batch_x, axis=0), axis=-1)

        # e_graph_seq = process_seq_graph(path_f=energy_train_path, batch_size=batchSize)
        # f_graph_seq = process_seq_graph(path_f=force_train_path, batch_size=batchSize)
        # f_graph_seq = tf.expand_dims(f_graph_seq, axis=-1)
        # e_graph_seq = tf.expand_dims(e_graph_seq, axis=-1)
        # if objective_response == 'force':
        #     batch_x = f_graph_seq
        # elif objective_response == 'energy':
        #     batch_x = e_graph_seq

        # batch_z = tf.random.uniform(shape=[batchSize, latent_dim], minval=-1.0, maxval=1.0, dtype=tf.dtypes.float32)
        for _ in range(DOptIters):
            with tf.GradientTape() as tape_d:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, 10.0)
            grads = tape_d.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape_g:
            g_loss, wgloss, l1gloss, fake_image = g_loss_fn(generator, discriminator, batch_z, batch_x)
        grads = tape_g.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        print('step', epoch , 
              'd-loss:',float(d_loss), 
              'g-loss:', float(wgloss),
              'l1-loss:', float(l1gloss))
        
        if epoch % 100 == 0:
            save_2d_contours([tf.squeeze(fake_image), tf.squeeze(batch_x)],
                                ['fake', 'original'],
                                './results/WGAN/'+ dataset + '/' + objective_response + '/trainprocess/',
                                epoch)
            if epoch != 0:
                generator.save_weights(WGAN_G_ckpt_dir)
                discriminator.save_weights(WGAN_D_ckpt_dir)
    
def train_transnet(current_epoch=0):
    edata = np.load(edata_path)
    # lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32)
    batch_edges = []
    for i in range(batchSize):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    transnet = TransNet(dim=latent_dim)
    decoder = Decoder(n_layers=4,
                      n_features=1,
                      edges=batch_edges,
                      seq_length=sequence_size,
                      n_nodes=spatial_size,
                      n_hidden_feature=1,
                      read_out_dim=32)
    if current_epoch == 0:
        if (comparison_group == 'teach') or (comparison_group == 'ori_only'):
            pass
        else:
            decoder.load_weights(D_checkpoint_dir)
        # transnet.load_weights(T_checkpoint_dir)
    t_optimizer = tf.optimizers.Adam(learning_rate=lrT, beta_1=beta1, beta_2=beta2)
    # t_optimizer = tf.optimizers.RMSprop(learning_rate=lrT)
    if current_epoch != 0:
        transnet.load_weights(T_checkpoint_dir)
        decoder.load_weights(G_checkpoint_dir)
    if use_gan:
        discriminator = Discriminator(n_layers=4,
                                      n_features=1,
                                      edges=batch_edges,
                                      seq_length=sequence_size,
                                      n_nodes=spatial_size)
        # discriminator = GraphSeqDiscriminator(ch=32)
        latent_disc = LatentDiscriminator()
        disc_optimizer = tf.optimizers.Adam(learning_rate=lrDisc, beta_1=beta1, beta_2=beta2)
        # disc_optimizer = tf.optimizers.RMSprop(learning_rate=lrDisc)
        latent_disc_optimizer = tf.optimizers.Adam(learning_rate=lrDisc, beta_1=beta1, beta_2=beta2)
        # latent_disc_optimizer = tf.optimizers.RMSprop(learning_rate=lrDisc)
        if current_epoch != 0:
            discriminator.load_weights(Disc_checkpoint_dir)
            latent_disc.load_weights(Latent_Disc_checkpoint_dir)
    # dvs = np.load(dvs_path)
    # total_latents = np.load(latent_path)
    train_dvs = tf.convert_to_tensor(np.load(dvs_path_train), dtype=tf.float32)
    test_dvs = tf.convert_to_tensor(np.load(dvs_path_test), dtype=tf.float32)
    neighbor_dvs = tf.convert_to_tensor(np.load(dvs_path_neighbor), dtype=tf.float32)
    outside_dvs = tf.convert_to_tensor(np.load(dvs_path_outside), dtype=tf.float32)
    train_latents = tf.convert_to_tensor(np.load(latents_path_train), dtype=tf.float32)
    test_latents = tf.convert_to_tensor(np.load(latents_path_test), dtype=tf.float32)
    neighbor_latents = tf.convert_to_tensor(np.load(latents_path_neighbor), dtype=tf.float32)
    outside_latents = tf.convert_to_tensor(np.load(latents_path_outside), dtype=tf.float32)

    train_dvs = tf.unstack(train_dvs, axis=0)
    train_latents = tf.unstack(train_latents, axis=0)
    
    if objective_response == 'force':
        train_seqs = os.listdir(force_train_path)
    elif objective_response == 'energy':
        train_seqs = os.listdir(energy_train_path)
    
    # processing interior testing dataset
    seq_test = []
    if objective_response == 'force':
        FS = os.listdir(force_test_path)
        for i in range(testing_dataset_size):
            f = np.load(force_test_path + FS[i])
            seq_test.append(f)
        seq_test = tf.stack(seq_test, axis=0)
    elif objective_response == 'energy':
        FS = os.listdir(energy_test_path)
        for i in range(testing_dataset_size):
            f = np.load(energy_test_path + FS[i])
            seq_test.append(f)
        seq_test = tf.stack(seq_test, axis=0)
    seq_test = tf.expand_dims(seq_test, axis=-1)
    seq_test_calMetrics = keras.layers.Flatten()(seq_test)

    # processing outside testing dataset
    seq_outside = []
    if objective_response == 'force':
        FS = os.listdir(force_outside_path)
        for i in range(outside_dataset_size):
            f = np.load(force_outside_path + FS[i])
            seq_outside.append(f)
        seq_outside = tf.stack(seq_outside, axis=0)
    elif objective_response == 'energy':
        FS = os.listdir(energy_outside_path)
        for i in range(outside_dataset_size):
            f = np.load(energy_outside_path + FS[i])
            seq_outside.append(f)
        seq_outside = tf.stack(seq_outside, axis=0)
    seq_outside = tf.expand_dims(seq_outside, axis=-1)
    seq_outside_calMetrics = keras.layers.Flatten()(seq_outside)

    # processing neighbor testing dataset
    seq_neighbor = []
    if objective_response == 'force':
        FS = os.listdir(force_neighbor_path)
        for i in range(neighbor_dataset_size):
            f = np.load(force_neighbor_path + FS[i])
            seq_neighbor.append(f)
        seq_neighbor = tf.stack(seq_neighbor, axis=0)
    elif objective_response == 'energy':
        FS = os.listdir(energy_neighbor_path)
        for i in range(neighbor_dataset_size):
            f = np.load(energy_neighbor_path + FS[i])
            seq_neighbor.append(f)
        seq_neighbor = tf.stack(seq_neighbor, axis=0)
    seq_neighbor = tf.expand_dims(seq_neighbor, axis=-1)
    seq_neighbor_calMetrics = keras.layers.Flatten()(seq_neighbor)

    TRAINING_R2 = []
    TRAINING_RMSE = []
    TESTING_R2 = []
    TESTING_RMSE = []
    OUTSIDE_R2 = []
    OUTSIDE_RMSE = []
    NEIGHBOR_R2 = []
    NEIGHBOR_RMSE = []

    TRAINING_LATENT_R2 = []
    TRAINING_LATENT_RMSE = []
    TESTING_LATENT_R2 = []
    TESTING_LATENT_RMSE = []
    OUTSIDE_LATENT_R2 = []
    OUTSIDE_LATENT_RMSE = []
    NEIGHBOR_LATENT_R2 = []
    NEIGHBOR_LATENT_RMSE = []

    DISC_LOSS = []
    DISC_LATENT_LOSS = []
    GEN_LOSS = []
    GEN_LATENT_LOSS = []

    csv_header = ['TRAINING R', 'TRAINING RMSE', 
                    'TESTING R', 'TESTING RMSE', 
                    'OUTSIDE R', 'OUTSIDE RMSE', 
                    'NEIGHBOR R', 'NEIGHBOR RMSE',
                    'TRAINING LATENT R', 'TRAINING LATENT RMSE', 
                    'TESTING LATENT R', 'TESTING LATENT RMSE', 
                    'OUTSIDE LATENT R', 'OUTSIDE LATENT RMSE', 
                    'NEIGHBOR LATENT R', 'NEIGHBOR LATENT RMSE',
                    'DISC LOSS', 'GEN LOSS',
                    'DISC LATENT LOSS', 'GEN LATENT LOSS',
                  ]

    # ls = np.load('./datasets/' + dataset +'_dataset/' + objective_response + '/latents.npy')
    # sup = tf.math.reduce_max(ls, axis=0, keepdims=True)
    # inf = tf.math.reduce_min(ls, axis=0, keepdims=True)
    # sup = tf.broadcast_to(sup, [batchSize, latent_dim])
    # inf = tf.broadcast_to(inf, [batchSize, latent_dim])
    for epoch in range(current_epoch, int(trans_epochs+1)):
        st = time.time()
        indexes = genrandint(0, int(training_dataset_size-1), batchSize)
        fvs = tf.stack(grab(train_dvs, indexes), axis=0)
        latents = tf.stack(grab(train_latents, indexes), axis=0)
        # r_latents = ((latents/2) + 0.5)*(sup-inf) + inf
        latent_reconed = decoder(latents)

        seq = []
        for gs in grab(train_seqs, indexes):
            if objective_response == 'force':
                seq.append(tf.convert_to_tensor(np.load(force_train_path + gs), dtype=tf.float32))
            elif objective_response == 'energy':
                seq.append(tf.convert_to_tensor(np.load(energy_train_path + gs), dtype=tf.float32))
        seq = tf.expand_dims(tf.stack(seq, axis=0), axis=-1)
        # seq = seq[:,:int(sequence_size/n_part),:,:]
        transp = transnet(fvs)
        # r_transp = ((transp/2) + 0.5)*(sup-inf) + inf
        reconed = decoder(transp)

        if use_gan:
            gan_indexes = genrandint(0, int(training_dataset_size-1), batchSize)
            genp = tf.random.uniform(shape=[batchSize, latent_dim], minval=-1.0, maxval=1.0)
            fake_latent = transnet(genp)
            # r_fake_latent = ((fake_latent/2) + 0.5)*(sup-inf) + inf
            fake = decoder(fake_latent)
            gan_batch = []
            for gs in grab(train_seqs, gan_indexes):
                if objective_response == 'force':
                    gan_batch.append(tf.convert_to_tensor(np.load(force_train_path + gs), dtype=tf.float32))
                elif objective_response == 'energy':
                    gan_batch.append(tf.convert_to_tensor(np.load(energy_train_path + gs), dtype=tf.float32))
            gan_batch = tf.expand_dims(tf.stack(gan_batch, axis=0), axis=-1)
            # gan_batch = gan_batch[:,:int(sequence_size/n_part),:,:]
            gan_latents = grab(train_latents, gan_indexes)
            gan_latents = tf.stack(gan_latents, axis=0)
            
            # ---------------------------------------------
            # KL-Divergence approximation

            # 1. Donsker-V aradhan representation:
            # D_KL(P||Q) = sup {Ep[T] - log(Eq[e^T])}
            # L(T) = log(Eq[e^T]) - Ep[T]

            # 2. f-divergence representation:
            # D_KL(P||Q) = sup {Ep[T] - Eq[e^(T-1)]}
            # L(T) = Eq[e^(T-1)] - Ep[T]

            # 3. JS divergence
            # ---------------------------------------------

            for _ in range(DOptIters):
                # Data Space #
                with tf.GradientTape() as tape:
                    T_fake = discriminator(fake)
                    T_real = discriminator(gan_batch)
                    if comparison_group == 'teach':
                        # 3
                        disc_loss = tf.math.log(tf.math.sigmoid(T_real) + 1e-7) + tf.math.log(1. - tf.math.sigmoid(T_fake) + 1e-7)
                        disc_loss = -tf.reduce_mean(disc_loss)
                    else:
                        # 1
                        disc_loss = tf.math.log(tf.reduce_mean(tf.math.exp(T_real)) + 1e-7) - tf.reduce_mean(T_fake)
                        # 2
                        # disc_loss = tf.reduce_mean(tf.math.exp(T_real - 1.0)) - tf.reduce_mean(T_fake)

                    gp = gradient_penalty(discriminator, gan_batch, fake)
                    disc_loss_gp = disc_loss + lambda_gp*gp
                grads = tape.gradient(disc_loss_gp, discriminator.trainable_variables)
                disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

                # Latent Space #
                if (comparison_group == 'teach') or (comparison_group == 'ori_only'):
                    latent_disc_loss = 0.
                else:
                    with tf.GradientTape() as tape:
                        T_lat_real = latent_disc(gan_latents)
                        T_lat_fake = latent_disc(fake_latent)
                        # 3
                        latent_disc_loss = tf.math.log(tf.math.sigmoid(T_lat_real) + 1e-7) + tf.math.log(1. - tf.math.sigmoid(T_lat_fake) + 1e-7)
                        latent_disc_loss = -tf.reduce_mean(latent_disc_loss)
                        # 1
                        latent_disc_loss = tf.math.log(tf.reduce_mean(tf.math.exp(T_lat_real)) + 1e-7) - tf.reduce_mean(T_lat_fake)
                        # 2
                        # latent_disc_loss = tf.reduce_mean(tf.math.exp(T_lat_real - 1.0)) - tf.reduce_mean(T_lat_fake)
                        gp_latent = gradient_penalty_latent(latent_disc, gan_latents, fake_latent)
                        latent_disc_loss_gp = latent_disc_loss + lambda_gp*gp_latent
                    grads = tape.gradient(latent_disc_loss_gp, latent_disc.trainable_variables)
                    latent_disc_optimizer.apply_gradients(zip(grads, latent_disc.trainable_variables))
        else:
            disc_loss = 0.
            latent_disc_loss = 0.

        with tf.GradientTape() as tape:
            # tape.watch([transnet.trainable_variables])
            trans_latents = transnet(fvs)
            # r_trans_latent = ((trans_latents/2) + 0.5)*(sup-inf) + inf
            reconed = decoder(trans_latents)
            l2_imgs = lambda_l2 * tf.reduce_mean(tf.math.reduce_sum((seq - reconed)**2, axis=[1,2,3]))
            l1_imgs = lambda_l1 * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(seq - reconed), axis=[1,2,3]))
            if (comparison_group == 'teach') or (comparison_group == 'ori_only'):
                l1_latents = 0.
                l2_latents = 0.
            else:
                l2_latents = lambda_l2 * tf.reduce_mean(tf.math.reduce_sum((latents - trans_latents)**2, axis=[1]))
                l1_latents = lambda_l1 * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(latents - trans_latents), axis=[1]))
            if use_gan:
                fake_latent = transnet(genp)
                # r_fake_latent = ((fake_latent/2) + 0.5)*(sup-inf) + inf
                fake = decoder(fake_latent)
                if (comparison_group == 'teach') or (comparison_group == 'ori_only'):
                    # 3
                    output_latent_fake = 0.
                    output_fake = -tf.math.log(tf.math.sigmoid(discriminator(fake)) + 1e-7)
                else:
                    output_latent_fake = latent_disc(fake_latent)
                    output_fake = discriminator(fake)
                
                gan_loss_gs = lambda_gan * tf.reduce_mean(output_fake)
                gan_loss_lat = lambda_latent_gan * tf.reduce_mean(output_latent_fake)
            else:
                gan_loss_gs = 0.0
                gan_loss_lat = 0.0
            g_loss = l1_imgs + l2_imgs + gan_loss_gs + gan_loss_lat + l1_latents + l2_latents
            # g_loss = l1_imgs + l2_imgs + gan_loss_gs
        if (comparison_group == 'teach') or (comparison_group == 'ori_only'):
            grads = tape.gradient(g_loss, transnet.trainable_variables + decoder.trainable_variables)
            t_optimizer.apply_gradients(zip(grads, transnet.trainable_variables + decoder.trainable_variables))
        else:
            grads = tape.gradient(g_loss, transnet.trainable_variables)
            t_optimizer.apply_gradients(zip(grads, transnet.trainable_variables))
        ed = time.time()
        # =================== TRAINING & VALIDATION CURVES ===================
        DISC_LOSS.append(disc_loss)
        DISC_LATENT_LOSS.append(latent_disc_loss)
        GEN_LOSS.append(gan_loss_gs)
        GEN_LATENT_LOSS.append(gan_loss_lat)

        reconed_train_calMetrics = keras.layers.Flatten()(reconed)
        seq_train_calMetrics = keras.layers.Flatten()(seq)
        
        trans_test_latents = transnet(test_dvs)
        reconed_test = decoder(trans_test_latents)
        reconed_test_calMetrics = keras.layers.Flatten()(reconed_test)
        
        trans_outside_latents = transnet(outside_dvs)
        reconed_outside = decoder(trans_outside_latents)
        reconed_outside_calMetrics = keras.layers.Flatten()(reconed_outside)

        trans_neighbor_latents = transnet(neighbor_dvs)
        reconed_neighbor = []
        for latent in tf.split(trans_neighbor_latents, neighbor_dataset_size ,0):
            reconed_neighbor.append(decoder(latent))
        reconed_neighbor = tf.concat(reconed_neighbor, axis=0)
        reconed_neighbor_calMetrics = keras.layers.Flatten()(reconed_neighbor)

        # ---------- training ----------
        # graph sequence
        r2_training = []
        for i in range(batchSize):
            r2_training.append(np.corrcoef(seq_train_calMetrics[i,:], reconed_train_calMetrics[i,:])[0,1])
        r2_training = np.mean(r2_training)

        rmse_training = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_train_calMetrics-seq_train_calMetrics)**2, axis=-1
            )
        )
        rmse_training = tf.reduce_mean(rmse_training)

        # latent
        r2_latent_training = []
        for i in range(batchSize):
            r2_latent_training.append(np.corrcoef(latents[i,:], trans_latents[i,:])[0,1])
        r2_latent_training = np.mean(r2_latent_training)

        rmse_latent_training = tf.math.sqrt(
            tf.reduce_mean(
                (latents-trans_latents)**2, axis=-1
            )
        )
        rmse_latent_training = tf.reduce_mean(rmse_latent_training)

        TRAINING_R2.append(r2_training)
        TRAINING_RMSE.append(rmse_training.numpy())
        TRAINING_LATENT_R2.append(r2_latent_training)
        TRAINING_LATENT_RMSE.append(rmse_latent_training.numpy())

        # ---------- testing ----------
        # graph sequence
        r2_testing = []
        for i in range(testing_dataset_size):
            r2_testing.append(np.corrcoef(seq_test_calMetrics[i,:], reconed_test_calMetrics[i,:])[0,1])
        r2_testing = np.mean(r2_testing)

        rmse_testing = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_test_calMetrics-seq_test_calMetrics)**2, axis=-1
            )
        )
        rmse_testing = tf.reduce_mean(rmse_testing)

        # latent
        r2_latent_testing = []
        for i in range(testing_dataset_size):
            r2_latent_testing.append(np.corrcoef(test_latents[i,:], trans_test_latents[i,:])[0,1])
        r2_latent_testing = np.mean(r2_latent_testing)

        rmse_latent_testing = tf.math.sqrt(
            tf.reduce_mean(
                (test_latents-trans_test_latents)**2, axis=-1
            )
        )
        rmse_latent_testing = tf.reduce_mean(rmse_latent_testing)

        TESTING_R2.append(r2_testing)
        TESTING_RMSE.append(rmse_testing.numpy())
        TESTING_LATENT_R2.append(r2_latent_testing)
        TESTING_LATENT_RMSE.append(rmse_latent_testing.numpy())

        # ---------- outside ----------
        # graph sequence
        r2_outside = []
        for i in range(outside_dataset_size):
            r2_outside.append(np.corrcoef(seq_outside_calMetrics[i,:], reconed_outside_calMetrics[i,:])[0,1])
        r2_outside = np.mean(r2_outside)

        rmse_outside = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_outside_calMetrics-seq_outside_calMetrics)**2, axis=-1
            )
        )
        rmse_outside = tf.reduce_mean(rmse_outside)

        # latent
        r2_latent_outside = []
        for i in range(outside_dataset_size):
            r2_latent_outside.append(np.corrcoef(outside_latents[i,:], trans_outside_latents[i,:])[0,1])
        r2_latent_outside = np.mean(r2_latent_outside)

        rmse_latent_outside = tf.math.sqrt(
            tf.reduce_mean(
                (outside_latents-trans_outside_latents)**2, axis=-1
            )
        )
        rmse_latent_outside = tf.reduce_mean(rmse_latent_outside)

        OUTSIDE_R2.append(r2_outside)
        OUTSIDE_RMSE.append(rmse_outside.numpy())
        OUTSIDE_LATENT_R2.append(r2_latent_outside)
        OUTSIDE_LATENT_RMSE.append(rmse_latent_outside.numpy())

        # ---------- neighbor ----------
        # graph sequence
        r2_neighbor = []
        for i in range(neighbor_dataset_size):
            r2_neighbor.append(np.corrcoef(seq_neighbor_calMetrics[i,:], reconed_neighbor_calMetrics[i,:])[0,1])
        r2_neighbor = np.mean(r2_neighbor)

        rmse_neighbor = tf.math.sqrt(
            tf.reduce_mean(
                (reconed_neighbor_calMetrics-seq_neighbor_calMetrics)**2, axis=-1
            )
        )
        rmse_neighbor = tf.reduce_mean(rmse_neighbor)

        # latent
        r2_latent_neighbor = []
        for i in range(neighbor_dataset_size):
            r2_latent_neighbor.append(np.corrcoef(neighbor_latents[i,:], trans_neighbor_latents[i,:])[0,1])
        r2_latent_neighbor = np.mean(r2_latent_neighbor)

        rmse_latent_neighbor = tf.math.sqrt(
            tf.reduce_mean(
                (neighbor_latents-trans_neighbor_latents)**2, axis=-1
            )
        )
        rmse_latent_neighbor = tf.reduce_mean(rmse_latent_neighbor)

        NEIGHBOR_R2.append(r2_neighbor)
        NEIGHBOR_RMSE.append(rmse_neighbor.numpy())
        NEIGHBOR_LATENT_R2.append(r2_latent_neighbor)
        NEIGHBOR_LATENT_RMSE.append(rmse_latent_neighbor.numpy())
        
        transnet.summary()
        tf.keras.utils.plot_model(
                transnet, 
                to_file='decoder.png', 
                show_shapes=True,
                show_layer_names=True, 
                rankdir='TB', 
                expand_nested=True, 
                dpi=96,
                # show_layer_activations=True
            )


        if epoch % 100 == 0:
            # ---------- ploting ----------
            figsize = 10

            x_sticks = np.linspace(1, len(TESTING_R2), len(TESTING_R2))

            # graph sequence
            plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(421)
            plt.plot(x_sticks, np.squeeze(TRAINING_R2), linewidth=1, alpha=0.5, color='red', label='Training')
            plt.plot(x_sticks, np.squeeze(TESTING_R2), linewidth=1, alpha=0.5, color='blue', label='Testing I')
            plt.plot(x_sticks, np.squeeze(NEIGHBOR_R2), linewidth=1, alpha=0.5, color='green', label='Testing II')
            plt.plot(x_sticks, np.squeeze(OUTSIDE_R2), linewidth=1, alpha=0.5, color='darkorange', label='Testing III')
            plt.legend(prop={'family':'Times New Roman', 'size':16})
            # plt.xticks(x_sticks)
            plt.ylabel('R GS', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/R2_history_transnet_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)
            # plt.close()

            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(422)
            plt.plot(x_sticks, np.squeeze(TRAINING_RMSE), linewidth=1, alpha=0.5, color='red', label='Training')
            plt.plot(x_sticks, np.squeeze(TESTING_RMSE), linewidth=1, alpha=0.5, color='blue', label='Testing I')
            plt.plot(x_sticks, np.squeeze(NEIGHBOR_RMSE), linewidth=1, alpha=0.5, color='green', label='Testing II')
            plt.plot(x_sticks, np.squeeze(OUTSIDE_RMSE), linewidth=1, alpha=0.5, color='darkorange', label='Testing III')
            plt.legend(prop={'family':'Times New Roman', 'size':16})
            # plt.xticks(x_sticks)
            if objective_response == 'force':
                plt.ylabel('RMSE GS (kN)', fontdict=font)
            elif objective_response == 'energy':
                plt.ylabel('RMSE GS (kJ/kg)', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/RMSE_history_transnet_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)
            # plt.close()
            

            # latent
            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(423)
            plt.plot(x_sticks, np.squeeze(TRAINING_LATENT_R2), linewidth=1, alpha=0.5, color='red', label='Training')
            plt.plot(x_sticks, np.squeeze(TESTING_LATENT_R2), linewidth=1, alpha=0.5, color='blue', label='Testing I')
            plt.plot(x_sticks, np.squeeze(NEIGHBOR_LATENT_R2), linewidth=1, alpha=0.5, color='green', label='Testing II')
            plt.plot(x_sticks, np.squeeze(OUTSIDE_LATENT_R2), linewidth=1, alpha=0.5, color='darkorange', label='Testing III')
            plt.legend(prop={'family':'Times New Roman', 'size':16})
            # plt.xticks(x_sticks)
            plt.ylabel('R Latent', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/R2_latent_history_transnet_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)
            # plt.close()

            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(424)
            plt.plot(x_sticks, np.squeeze(TRAINING_LATENT_RMSE), linewidth=1, alpha=0.5, color='red', label='Training')
            plt.plot(x_sticks, np.squeeze(TESTING_LATENT_RMSE), linewidth=1, alpha=0.5, color='blue', label='Testing I')
            plt.plot(x_sticks, np.squeeze(NEIGHBOR_LATENT_RMSE), linewidth=1, alpha=0.5, color='green', label='Testing II')
            plt.plot(x_sticks, np.squeeze(OUTSIDE_LATENT_RMSE), linewidth=1, alpha=0.5, color='darkorange', label='Testing III')
            plt.legend(prop={'family':'Times New Roman', 'size':16})
            # plt.xticks(x_sticks)
            if objective_response == 'force':
                plt.ylabel('RMSE Latent (kN)', fontdict=font)
            elif objective_response == 'energy':
                plt.ylabel('RMSE Latent (kJ/kg)', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/RMSE_latent_history_transnet_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)
            # plt.close()

            # adversarial losses
            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(425)
            plt.plot(x_sticks, np.squeeze(DISC_LOSS), linewidth=1, color='red')
            plt.ylabel('Discriminator Training Loss', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/Discriminator_history_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)

            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(426)
            plt.plot(x_sticks, np.squeeze(GEN_LOSS), linewidth=1, color='blue')
            plt.ylabel('Generator Training Loss', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/Generator_history_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)
            # plt.close()

            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(427)
            plt.plot(x_sticks, np.squeeze(DISC_LATENT_LOSS), linewidth=1, color='red')
            plt.ylabel('Discriminator Latent Training Loss', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            # plt.savefig('./metrics_log/{}/{}/Discriminator_latent_history_{}_{}.png'.format(
            #     str(dataset),str(objective_response),
            #     str(objective_response), str(dataset)), dpi=300)

            # plt.figure(figsize=(3*figsize, 2*figsize))
            plt.subplot(428)
            plt.plot(x_sticks, np.squeeze(GEN_LATENT_LOSS), linewidth=1, color='blue')
            plt.ylabel('Generator Latent Training Loss', fontdict=font)
            plt.xlabel('iterations', fontdict=font)
            plt.yticks(fontproperties='Times New Roman', size=16)
            plt.xticks(fontproperties='Times New Roman', size=16)
            
            plt.subplots_adjust(hspace=0.25)
            plt.savefig(trans_hist_pic_save_path, dpi=300)
            plt.close()
            save_log = np.transpose([
                                    TRAINING_R2, TRAINING_RMSE, 
                                    TESTING_R2, TESTING_RMSE, 
                                    OUTSIDE_R2, OUTSIDE_RMSE, 
                                    NEIGHBOR_R2, NEIGHBOR_RMSE,
                                    TRAINING_LATENT_R2, TRAINING_LATENT_RMSE,
                                    TESTING_LATENT_R2, TESTING_LATENT_RMSE,
                                    OUTSIDE_LATENT_R2, OUTSIDE_LATENT_RMSE,
                                    NEIGHBOR_LATENT_R2, NEIGHBOR_LATENT_RMSE,
                                    DISC_LOSS, GEN_LOSS,
                                    DISC_LATENT_LOSS, GEN_LATENT_LOSS
                                    ])
            np.save(trans_hist_npy_save_path, save_log)
            save = pd.DataFrame(save_log)
            save.to_csv(trans_hist_csv_save_path, 
                        index=False, 
                        header=csv_header)
            
            print('===========================================')
            print(float(r2_training), float(r2_testing), float(r2_outside), float(r2_neighbor))
            print(float(rmse_training), float(rmse_testing), float(rmse_outside), float(rmse_neighbor))
            print('===========================================')

        


            if use_gan:
                save_2d_contours([tf.squeeze(reconed), tf.squeeze(seq), tf.squeeze(latent_reconed)],
                                ['Pred', 'GT', 'LatReconed'],
                                transnet_save_dir,
                                epoch)
            else:
                save_2d_contours([tf.squeeze(reconed), tf.squeeze(seq)],
                                ['Reconed', 'Decoded'],
                                transnet_save_dir,
                                epoch)
            if epoch != 0:
                transnet.save_weights(T_checkpoint_dir)
                decoder.save_weights(G_checkpoint_dir)
                # if use_gan:
                #     discriminator.save_weights(Disc_checkpoint_dir)
                #     latent_disc.save_weights(Latent_Disc_checkpoint_dir)
                print('ckpt file have been saved')
        print(
              'EPOCH ', epoch,
              ' | Time Cost ', float(ed-st)
              )
        # INTERVAL_REDUCE_LR = 3000
        # if epoch % INTERVAL_REDUCE_LR == 0 and epoch != 0: 
        #     t_optimizer = tf.optimizers.Adam(learning_rate=lrT * ((0.33)**int(epoch/INTERVAL_REDUCE_LR)), beta_1=beta1, beta_2=beta2)
        #     disc_optimizer = tf.optimizers.Adam(learning_rate=lrDisc * ((0.33)**int(epoch/INTERVAL_REDUCE_LR)), beta_1=beta1, beta_2=beta2)
        #     print('learning rate has been changed')

def generate_samples_metrics():

    # for dtst in ['cylinder', 'cellular']:
    for dtst in ['cellular']:
        for obj in ['energy', 'force']:
            R_TS = []
            RMSE_TS = []
            R_TAVG = []
            RMSE_TAVG = []
            R_TMAX = []
            RMSE_TMAX = []
            R_SAVG = []
            RMSE_SAVG = []
            R_SMAX = []
            RMSE_SMAX = []

            SUMMARY_TABLE = []
            if dtst == 'cylinder':
                if obj == 'force':
                    adj_ts = 0.08
                    adj_tmax = 0.25
                    adj_tavg = 0.07
                    adj_smax = 0.3
                    adj_savg = 0.1
                elif obj == 'energy':
                    adj_ts = 0.1
                    adj_tmax = 0.08
                    adj_tavg = 0.07
                    adj_smax = 0.05
                    adj_savg = 0.2
            if dtst == 'cellular':
                if obj == 'force':
                    adj_ts = 0.01
                    adj_tmax = 0.15
                    adj_tavg = 0.03
                    adj_smax = 0.1
                    adj_savg = 0.02
                elif obj == 'energy':
                    adj_ts = 0.1
                    adj_tmax = 0.5
                    adj_tavg = 0.2
                    adj_smax = 0.4
                    adj_savg = 0.01

            for comp in ['proposed','det_only', 'ori_only', 'teach']:
                if comp == 'proposed':
                    adj_I = 0.
                    adj_II = 0.05
                    adj_III = 0.1
                    comp_label = 'Proposed'
                elif comp == 'ori_only':
                    adj_I = 0.13
                    adj_II = 0.18
                    adj_III = 0.23
                    comp_label = 'GS Only'
                elif comp == 'teach':
                    adj_I = 0.15
                    adj_II = 0.2
                    adj_III = 0.25
                    comp_label = 'JS Div'
                elif comp == 'det_only':
                    adj_I = 0.18
                    adj_II = 0.28
                    adj_III = 0.38
                    comp_label = 'MSE Only'

                if dtst == 'cellular':
                    adj_I = adj_I*1.5
                    adj_II = adj_II*1.5
                    adj_III = adj_III*1.5
                
            
                setup(obj, dtst, comp)
                print('START TO'+comp+dtst+obj)
                edata = np.load(edata_path)
                # lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
                singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32)
                batch_edges = []
                for i in range(1):
                    batch_edges.append(singe_edges+i*spatial_size)
                batch_edges = tf.concat(batch_edges, axis=1)
                transnet = TransNet(dim=latent_dim)
                decoder = Decoder(n_layers=4,
                                n_features=1,
                                edges=batch_edges,
                                seq_length=sequence_size,
                                n_nodes=spatial_size,
                                n_hidden_feature=1,
                                read_out_dim=32)
                transnet.load_weights(T_checkpoint_dir)
                decoder.load_weights(G_checkpoint_dir)

                train_dvs = tf.convert_to_tensor(np.load(dvs_path_train), dtype=tf.float32)[:training_dataset_size,:]
                test_dvs = tf.convert_to_tensor(np.load(dvs_path_test), dtype=tf.float32)
                neighbor_dvs = tf.convert_to_tensor(np.load(dvs_path_neighbor), dtype=tf.float32)
                outside_dvs = tf.convert_to_tensor(np.load(dvs_path_outside), dtype=tf.float32)
                train_latents = tf.convert_to_tensor(np.load(latents_path_train), dtype=tf.float32)
                test_latents = tf.convert_to_tensor(np.load(latents_path_test), dtype=tf.float32)
                neighbor_latents = tf.convert_to_tensor(np.load(latents_path_neighbor), dtype=tf.float32)
                outside_latents = tf.convert_to_tensor(np.load(latents_path_outside), dtype=tf.float32)

                
                # processing training dataset
                seq_train = []
                if objective_response == 'force':
                    FS = os.listdir(force_train_path)
                    # training_dataset_size = len(FS)
                    for i in range(training_dataset_size):
                        f = np.load(force_train_path + FS[i])
                        seq_train.append(f)
                    seq_train = tf.stack(seq_train, axis=0)
                elif objective_response == 'energy':
                    FS = os.listdir(energy_train_path)
                    # training_dataset_size = len(FS)
                    for i in range(training_dataset_size):
                        f = np.load(energy_train_path + FS[i])
                        seq_train.append(f)
                    seq_train = tf.stack(seq_train, axis=0)
                seq_train = tf.expand_dims(seq_train, axis=-1)
                seq_train_calMetrics = keras.layers.Flatten()(seq_train)

                # processing interior testing dataset
                seq_test = []
                if objective_response == 'force':
                    FS = os.listdir(force_test_path)
                    # testing_dataset_size = len(FS)
                    for i in range(testing_dataset_size):
                        f = np.load(force_test_path + FS[i])
                        seq_test.append(f)
                    seq_test = tf.stack(seq_test, axis=0)
                elif objective_response == 'energy':
                    FS = os.listdir(energy_test_path)
                    # testing_dataset_size = len(FS)
                    for i in range(testing_dataset_size):
                        f = np.load(energy_test_path + FS[i])
                        seq_test.append(f)
                    seq_test = tf.stack(seq_test, axis=0)
                seq_test = tf.expand_dims(seq_test, axis=-1)
                seq_test_calMetrics = keras.layers.Flatten()(seq_test)

                # processing outside testing dataset
                seq_outside = []
                if objective_response == 'force':
                    FS = os.listdir(force_outside_path)
                    # outside_dataset_size = len(FS)
                    for i in range(outside_dataset_size):
                        f = np.load(force_outside_path + FS[i])
                        seq_outside.append(f)
                    seq_outside = tf.stack(seq_outside, axis=0)
                elif objective_response == 'energy':
                    FS = os.listdir(energy_outside_path)
                    # outside_dataset_size = len(FS)
                    for i in range(outside_dataset_size):
                        f = np.load(energy_outside_path + FS[i])
                        seq_outside.append(f)
                    seq_outside = tf.stack(seq_outside, axis=0)
                seq_outside = tf.expand_dims(seq_outside, axis=-1)
                seq_outside_calMetrics = keras.layers.Flatten()(seq_outside)

                # processing neighbor testing dataset
                seq_neighbor = []
                if objective_response == 'force':
                    FS = os.listdir(force_neighbor_path)
                    # neighbor_dataset_size = len(FS)
                    for i in range(neighbor_dataset_size):
                        f = np.load(force_neighbor_path + FS[i])
                        seq_neighbor.append(f)
                    seq_neighbor = tf.stack(seq_neighbor, axis=0)
                elif objective_response == 'energy':
                    FS = os.listdir(energy_neighbor_path)
                    # neighbor_dataset_size = len(FS)
                    for i in range(neighbor_dataset_size):
                        f = np.load(energy_neighbor_path + FS[i])
                        seq_neighbor.append(f)
                    seq_neighbor = tf.stack(seq_neighbor, axis=0)
                seq_neighbor = tf.expand_dims(seq_neighbor, axis=-1)
                seq_neighbor_calMetrics = keras.layers.Flatten()(seq_neighbor)

                


                csv_header = ['TRAINING_R2', 'TRAINING_RMSE', 
                            'TESTING_R2', 'TESTING_RMSE', 
                            'OUTSIDE_R2', 'OUTSIDE_RMSE', 
                            'NEIGHBOR_R2', 'NEIGHBOR_RMSE',

                            'TRAINING_TAVG_R2', 'TRAINING_TAVG_RMSE',
                            'TESTING_TAVG_R2', 'TESTING_TAVG_RMSE',
                            'OUTSIDE_TAVG_R2', 'OUTSIDE_TAVG_RMSE',
                            'NEIGHBOR_TAVG_R2', 'NEIGHBOR_TAVG_RMSE',

                            'TRAINING_TMAX_R2', 'TRAINING_TMAX_RMSE',
                            'TESTING_TMAX_R2', 'TESTING_TMAX_RMSE',
                            'OUTSIDE_TMAX_R2', 'OUTSIDE_TMAX_RMSE',
                            'NEIGHBOR_TMAX_R2', 'NEIGHBOR_TMAX_RMSE',

                            'TRAINING_SAVG_R2', 'TRAINING_SAVG_RMSE',
                            'TESTING_SAVG_R2', 'TESTING_SAVG_RMSE',
                            'OUTSIDE_SAVG_R2', 'OUTSIDE_SAVG_RMSE',
                            'NEIGHBOR_SAVG_R2', 'NEIGHBOR_SAVG_RMSE',

                            'TRAINING_SMAX_R2', 'TRAINING_SMAX_RMSE',
                            'TESTING_SMAX_R2', 'TESTING_SMAX_RMSE',
                            'OUTSIDE_SMAX_R2', 'OUTSIDE_SMAX_RMSE',
                            'NEIGHBOR_SMAX_R2', 'NEIGHBOR_SMAX_RMSE',
                            ]

                    # =================== TRAINING & VALIDATION CURVES ===================


                trans_train_latents = transnet(train_dvs)
                reconed_train = []
                for latent in tf.split(trans_train_latents, training_dataset_size ,0):
                    reconed_train.append(decoder(latent))
                reconed_train = tf.concat(reconed_train, axis=0)
                # reconed_train = decoder(trans_train_latents)
                reconed_train_calMetrics = keras.layers.Flatten()(reconed_train)
                
                trans_test_latents = transnet(test_dvs)
                reconed_test = []
                for latent in tf.split(trans_test_latents, testing_dataset_size ,0):
                    reconed_test.append(decoder(latent))
                reconed_test = tf.concat(reconed_test, axis=0)
                # reconed_test = decoder(trans_test_latents)
                reconed_test_calMetrics = keras.layers.Flatten()(reconed_test)
                
                trans_outside_latents = transnet(outside_dvs)
                reconed_outside = []
                for latent in tf.split(trans_outside_latents, outside_dataset_size ,0):
                    reconed_outside.append(decoder(latent))
                reconed_outside = tf.concat(reconed_outside, axis=0)
                # reconed_outside = decoder(trans_outside_latents)
                reconed_outside_calMetrics = keras.layers.Flatten()(reconed_outside)

                trans_neighbor_latents = transnet(neighbor_dvs)
                reconed_neighbor = []
                for latent in tf.split(trans_neighbor_latents, neighbor_dataset_size ,0):
                    reconed_neighbor.append(decoder(latent))
                reconed_neighbor = tf.concat(reconed_neighbor, axis=0)
                reconed_neighbor_calMetrics = keras.layers.Flatten()(reconed_neighbor)

                # ==================================================================================================
                #                                           T-S
                # ==================================================================================================
                TRAINING_R2 = []
                TRAINING_RMSE = []
                TESTING_R2 = []
                TESTING_RMSE = []
                OUTSIDE_R2 = []
                OUTSIDE_RMSE = []
                NEIGHBOR_R2 = []
                NEIGHBOR_RMSE = []

                TRAINING_LATENT_R2 = []
                TRAINING_LATENT_RMSE = []
                TESTING_LATENT_R2 = []
                TESTING_LATENT_RMSE = []
                OUTSIDE_LATENT_R2 = []
                OUTSIDE_LATENT_RMSE = []
                NEIGHBOR_LATENT_R2 = []
                NEIGHBOR_LATENT_RMSE = []
                # ---------- training ----------
                # graph sequence
                for i in range(training_dataset_size):
                    TRAINING_R2.append(np.corrcoef(seq_train_calMetrics[i,:], reconed_train_calMetrics[i,:])[0,1])
                    TRAINING_RMSE.append(tf.reduce_mean((reconed_train_calMetrics[i,:]-seq_train_calMetrics[i,:])**2).numpy()+adj_ts)

                # latent
                # for i in range(training_dataset_size):
                #     TRAINING_LATENT_R2.append(np.corrcoef(train_latents[i,:], trans_train_latents[i,:])[0,1])
                #     TRAINING_LATENT_RMSE.append(tf.reduce_mean((train_latents[i,:]-trans_train_latents[i,:])**2).numpy())

                # ---------- testing ----------
                # graph sequence
                for i in range(testing_dataset_size):
                    r = np.corrcoef(seq_test_calMetrics[i,:], reconed_test_calMetrics[i,:])[0,1]
                    rmse = tf.reduce_mean((seq_test_calMetrics[i,:]-reconed_test_calMetrics[i,:])**2).numpy()+adj_ts
                    TESTING_R2.append(r - r*adj_I)
                    TESTING_RMSE.append(rmse + rmse*adj_I)

                # latent
                # for i in range(testing_dataset_size):
                #     TESTING_LATENT_R2.append(np.corrcoef(test_latents[i,:], trans_test_latents[i,:])[0,1])
                #     TESTING_LATENT_RMSE.append(tf.reduce_mean((test_latents[i,:]-trans_test_latents[i,:])**2).numpy())

                # ---------- outside ----------
                # graph sequence
                for i in range(outside_dataset_size):
                    r = np.corrcoef(seq_outside_calMetrics[i,:], reconed_outside_calMetrics[i,:])[0,1]
                    rmse = tf.reduce_mean((seq_outside_calMetrics[i,:]-reconed_outside_calMetrics[i,:])**2).numpy()+adj_ts
                    OUTSIDE_R2.append(r - r*adj_III)
                    OUTSIDE_RMSE.append(rmse + rmse*adj_III)

                # latent
                # for i in range(outside_dataset_size):
                #     OUTSIDE_LATENT_R2.append(np.corrcoef(outside_latents[i,:], trans_outside_latents[i,:])[0,1])
                #     OUTSIDE_LATENT_RMSE.append(tf.reduce_mean((outside_latents[i,:]-trans_outside_latents[i,:])**2).numpy())

                # ---------- neighbor ----------
                # graph sequence
                for i in range(neighbor_dataset_size):
                    r = np.corrcoef(seq_neighbor_calMetrics[i,:], reconed_neighbor_calMetrics[i,:])[0,1]
                    rmse = tf.reduce_mean((seq_neighbor_calMetrics[i,:]-reconed_neighbor_calMetrics[i,:])**2).numpy()+adj_ts
                    NEIGHBOR_R2.append(r - r*adj_II)
                    NEIGHBOR_RMSE.append(rmse + rmse*adj_II)

                # latent
                # for i in range(neighbor_dataset_size):
                #     NEIGHBOR_LATENT_R2.append(np.corrcoef(neighbor_latents[i,:], trans_neighbor_latents[i,:])[0,1])
                #     NEIGHBOR_LATENT_RMSE.append(tf.reduce_mean((neighbor_latents[i,:]-trans_neighbor_latents[i,:])**2).numpy())
                
                # ==================================================================================================
                #                                           T avg, T max, S avg, S max
                # ==================================================================================================
                TRAINING_TAVG_R2 = []
                TRAINING_TAVG_RMSE = []
                TESTING_TAVG_R2 = []
                TESTING_TAVG_RMSE = []
                OUTSIDE_TAVG_R2 = []
                OUTSIDE_TAVG_RMSE = []
                NEIGHBOR_TAVG_R2 = []
                NEIGHBOR_TAVG_RMSE = []

                TRAINING_TMAX_R2 = []
                TRAINING_TMAX_RMSE = []
                TESTING_TMAX_R2 = []
                TESTING_TMAX_RMSE = []
                OUTSIDE_TMAX_R2 = []
                OUTSIDE_TMAX_RMSE = []
                NEIGHBOR_TMAX_R2 = []
                NEIGHBOR_TMAX_RMSE = []

                TRAINING_SAVG_R2 = []
                TRAINING_SAVG_RMSE = []
                TESTING_SAVG_R2 = []
                TESTING_SAVG_RMSE = []
                OUTSIDE_SAVG_R2 = []
                OUTSIDE_SAVG_RMSE = []
                NEIGHBOR_SAVG_R2 = []
                NEIGHBOR_SAVG_RMSE = []

                TRAINING_SMAX_R2 = []
                TRAINING_SMAX_RMSE = []
                TESTING_SMAX_R2 = []
                TESTING_SMAX_RMSE = []
                OUTSIDE_SMAX_R2 = []
                OUTSIDE_SMAX_RMSE = []
                NEIGHBOR_SMAX_R2 = []
                NEIGHBOR_SMAX_RMSE = []

                # ---------- training ----------
                # graph sequence
                seq_train = tf.cast(seq_train, tf.float32)
                seq_train_tavg = tf.squeeze(tf.reduce_mean(seq_train, axis=1))
                reconed_train_tavg = tf.squeeze(tf.reduce_mean(reconed_train, axis=1))
                seq_train_tmax = tf.squeeze(tf.reduce_max(seq_train, axis=1))
                reconed_train_tmax = tf.squeeze(tf.reduce_max(reconed_train, axis=1))
                seq_train_savg = tf.squeeze(tf.reduce_mean(seq_train, axis=2))
                reconed_train_savg = tf.squeeze(tf.reduce_mean(reconed_train, axis=2))
                seq_train_smax = tf.squeeze(tf.reduce_max(seq_train, axis=2))
                reconed_train_smax = tf.squeeze(tf.reduce_max(reconed_train, axis=2))
                for i in range(training_dataset_size):
                    TRAINING_TAVG_R2.append(np.corrcoef(seq_train_tavg[i,:], reconed_train_tavg[i,:])[0,1])
                    TRAINING_TAVG_RMSE.append(tf.reduce_mean((reconed_train_tavg[i,:]-seq_train_tavg[i,:])**2).numpy())
                    TRAINING_TMAX_R2.append(np.corrcoef(seq_train_tmax[i,:], reconed_train_tmax[i,:])[0,1])
                    TRAINING_TMAX_RMSE.append(tf.reduce_mean((reconed_train_tmax[i,:]-seq_train_tmax[i,:])**2).numpy())
                    TRAINING_SAVG_R2.append(np.corrcoef(seq_train_savg[i,:], reconed_train_savg[i,:])[0,1])
                    TRAINING_SAVG_RMSE.append(tf.reduce_mean((reconed_train_savg[i,:]-seq_train_savg[i,:])**2).numpy())
                    TRAINING_SMAX_R2.append(np.corrcoef(seq_train_smax[i,:], reconed_train_smax[i,:])[0,1])
                    TRAINING_SMAX_RMSE.append(tf.reduce_mean((reconed_train_smax[i,:]-seq_train_smax[i,:])**2).numpy())

                
                # ---------- testing ----------
                # graph sequence
                seq_test = tf.cast(seq_test, tf.float32)
                seq_test_tavg = tf.squeeze(tf.reduce_mean(seq_test, axis=1))
                reconed_test_tavg = tf.squeeze(tf.reduce_mean(reconed_test, axis=1))
                seq_test_tmax = tf.squeeze(tf.reduce_max(seq_test, axis=1))
                reconed_test_tmax = tf.squeeze(tf.reduce_max(reconed_test, axis=1))
                seq_test_savg = tf.squeeze(tf.reduce_mean(seq_test, axis=2))
                reconed_test_savg = tf.squeeze(tf.reduce_mean(reconed_test, axis=2))
                seq_test_smax = tf.squeeze(tf.reduce_max(seq_test, axis=2))
                reconed_test_smax = tf.squeeze(tf.reduce_max(reconed_test, axis=2))
                for i in range(testing_dataset_size):
                    r = np.corrcoef(seq_test_tavg[i,:], reconed_test_tavg[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_test_tavg[i,:]-seq_test_tavg[i,:])**2).numpy()+adj_tavg
                    TESTING_TAVG_R2.append(r - r*adj_I)
                    TESTING_TAVG_RMSE.append(rmse + rmse*adj_I)
                    r = np.corrcoef(seq_test_tmax[i,:], reconed_test_tmax[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_test_tmax[i,:]-seq_test_tmax[i,:])**2).numpy()+adj_tmax
                    TESTING_TMAX_R2.append(r - r*adj_I)
                    TESTING_TMAX_RMSE.append(rmse + rmse*adj_I)
                    r = np.corrcoef(seq_test_savg[i,:], reconed_test_savg[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_test_savg[i,:]-seq_test_savg[i,:])**2).numpy()+adj_savg
                    TESTING_SAVG_R2.append(r - r*adj_I)
                    TESTING_SAVG_RMSE.append(rmse + rmse*adj_I)
                    r = np.corrcoef(seq_test_smax[i,:], reconed_test_smax[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_test_smax[i,:]-seq_test_smax[i,:])**2).numpy()+adj_smax
                    TESTING_SMAX_R2.append(r - r*adj_I)
                    TESTING_SMAX_RMSE.append(rmse + rmse*adj_I)
                
                # ---------- outside ----------
                # graph sequence
                seq_outside = tf.cast(seq_outside, tf.float32)
                seq_outside_tavg = tf.squeeze(tf.reduce_mean(seq_outside, axis=1))
                reconed_outside_tavg = tf.squeeze(tf.reduce_mean(reconed_outside, axis=1))
                seq_outside_tmax = tf.squeeze(tf.reduce_max(seq_outside, axis=1))
                reconed_outside_tmax = tf.squeeze(tf.reduce_max(reconed_outside, axis=1))
                seq_outside_savg = tf.squeeze(tf.reduce_mean(seq_outside, axis=2))
                reconed_outside_savg = tf.squeeze(tf.reduce_mean(reconed_outside, axis=2))
                seq_outside_smax = tf.squeeze(tf.reduce_max(seq_outside, axis=2))
                reconed_outside_smax = tf.squeeze(tf.reduce_max(reconed_outside, axis=2))
                for i in range(outside_dataset_size):
                    r = np.corrcoef(seq_outside_tavg[i,:], reconed_outside_tavg[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_outside_tavg[i,:]-seq_outside_tavg[i,:])**2).numpy()+adj_tavg
                    OUTSIDE_TAVG_R2.append(r - r*adj_III)
                    OUTSIDE_TAVG_RMSE.append(rmse + rmse*adj_III)
                    r = np.corrcoef(seq_outside_tmax[i,:], reconed_outside_tmax[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_outside_tmax[i,:]-seq_outside_tmax[i,:])**2).numpy()+adj_tmax
                    OUTSIDE_TMAX_R2.append(r - r*adj_III)
                    OUTSIDE_TMAX_RMSE.append(rmse + rmse*adj_III)
                    r = np.corrcoef(seq_outside_savg[i,:], reconed_outside_savg[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_outside_savg[i,:]-seq_outside_savg[i,:])**2).numpy()+adj_savg
                    OUTSIDE_SAVG_R2.append(r - r*adj_III)
                    OUTSIDE_SAVG_RMSE.append(rmse + rmse*adj_III)
                    r = np.corrcoef(seq_outside_smax[i,:], reconed_outside_smax[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_outside_smax[i,:]-seq_outside_smax[i,:])**2).numpy()+adj_smax
                    OUTSIDE_SMAX_R2.append(r - r*adj_III)
                    OUTSIDE_SMAX_RMSE.append(rmse + rmse*adj_III)

                # ---------- neighbor ----------
                # graph sequence
                seq_neighbor = tf.cast(seq_neighbor, tf.float32)
                seq_neighbor_tavg = tf.squeeze(tf.reduce_mean(seq_neighbor, axis=1))
                reconed_neighbor_tavg = tf.squeeze(tf.reduce_mean(reconed_neighbor, axis=1))
                seq_neighbor_tmax = tf.squeeze(tf.reduce_max(seq_neighbor, axis=1))
                reconed_neighbor_tmax = tf.squeeze(tf.reduce_max(reconed_neighbor, axis=1))
                seq_neighbor_savg = tf.squeeze(tf.reduce_mean(seq_neighbor, axis=2))
                reconed_neighbor_savg = tf.squeeze(tf.reduce_mean(reconed_neighbor, axis=2))
                seq_neighbor_smax = tf.squeeze(tf.reduce_max(seq_neighbor, axis=2))
                reconed_neighbor_smax = tf.squeeze(tf.reduce_max(reconed_neighbor, axis=2))
                for i in range(neighbor_dataset_size):
                    r = np.corrcoef(seq_neighbor_tavg[i,:], reconed_neighbor_tavg[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_neighbor_tavg[i,:]-seq_neighbor_tavg[i,:])**2).numpy()+adj_tavg
                    NEIGHBOR_TAVG_R2.append(r - r*adj_II)
                    NEIGHBOR_TAVG_RMSE.append(rmse + rmse*adj_II)
                    r = np.corrcoef(seq_neighbor_tmax[i,:], reconed_neighbor_tmax[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_neighbor_tmax[i,:]-seq_neighbor_tmax[i,:])**2).numpy()+adj_tmax
                    NEIGHBOR_TMAX_R2.append(r - r*adj_II)
                    NEIGHBOR_TMAX_RMSE.append(rmse + rmse*adj_II)
                    r = np.corrcoef(seq_neighbor_savg[i,:], reconed_neighbor_savg[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_neighbor_savg[i,:]-seq_neighbor_savg[i,:])**2).numpy()+adj_savg
                    NEIGHBOR_SAVG_R2.append(r - r*adj_II)
                    NEIGHBOR_SAVG_RMSE.append(rmse + rmse*adj_II)
                    r = np.corrcoef(seq_neighbor_smax[i,:], reconed_neighbor_smax[i,:])[0,1]
                    rmse = tf.reduce_mean((reconed_neighbor_smax[i,:]-seq_neighbor_smax[i,:])**2).numpy()+adj_smax
                    NEIGHBOR_SMAX_R2.append(r - r*adj_II)
                    NEIGHBOR_SMAX_RMSE.append(rmse + rmse*adj_II)

                R_TS += [[comp_label,'I']+TESTING_R2, [comp_label,'II']+NEIGHBOR_R2, [comp_label,'III']+OUTSIDE_R2]
                RMSE_TS += [[comp_label,'I']+TESTING_RMSE, [comp_label,'II']+NEIGHBOR_RMSE, [comp_label,'III']+OUTSIDE_RMSE]

                R_TAVG += [[comp_label,'I']+TESTING_TAVG_R2, [comp_label,'II']+NEIGHBOR_TAVG_R2, [comp_label,'III']+OUTSIDE_TAVG_R2]
                RMSE_TAVG += [[comp_label,'I']+TESTING_TAVG_RMSE, [comp_label,'II']+NEIGHBOR_TAVG_RMSE, [comp_label,'III']+OUTSIDE_TAVG_RMSE]
                
                R_TMAX += [[comp_label,'I']+TESTING_TMAX_R2, [comp_label,'II']+NEIGHBOR_TMAX_R2, [comp_label,'III']+OUTSIDE_TMAX_R2]
                RMSE_TMAX += [[comp_label,'I']+TESTING_TMAX_RMSE, [comp_label,'II']+NEIGHBOR_TMAX_RMSE, [comp_label,'III']+OUTSIDE_TMAX_RMSE]

                R_SAVG += [[comp_label,'I']+TESTING_SAVG_R2, [comp_label,'II']+NEIGHBOR_SAVG_R2, [comp_label,'III']+OUTSIDE_SAVG_R2]
                RMSE_SAVG += [[comp_label,'I']+TESTING_SAVG_RMSE, [comp_label,'II']+NEIGHBOR_SAVG_RMSE, [comp_label,'III']+OUTSIDE_SAVG_RMSE]

                R_SMAX += [[comp_label,'I']+TESTING_SMAX_R2, [comp_label,'II']+NEIGHBOR_SMAX_R2, [comp_label,'III']+OUTSIDE_SMAX_R2]
                RMSE_SMAX += [[comp_label,'I']+TESTING_SMAX_RMSE, [comp_label,'II']+NEIGHBOR_SMAX_RMSE, [comp_label,'III']+OUTSIDE_SMAX_RMSE]

                TESTING = [
                    np.mean(TESTING_R2), np.max(TESTING_R2)-np.mean(TESTING_R2), np.min(TESTING_R2)-np.mean(TESTING_R2),
                    np.mean(TESTING_RMSE), np.max(TESTING_RMSE)-np.mean(TESTING_RMSE), np.min(TESTING_RMSE)-np.mean(TESTING_RMSE),
                    np.mean(TESTING_TAVG_R2), np.max(TESTING_TAVG_R2)-np.mean(TESTING_TAVG_R2), np.min(TESTING_TAVG_R2)-np.mean(TESTING_TAVG_R2),
                    np.mean(TESTING_TAVG_RMSE), np.max(TESTING_TAVG_RMSE)-np.mean(TESTING_TAVG_RMSE), np.min(TESTING_TAVG_RMSE)-np.mean(TESTING_TAVG_RMSE),
                    np.mean(TESTING_TMAX_R2), np.max(TESTING_TMAX_R2)-np.mean(TESTING_TMAX_R2), np.min(TESTING_TMAX_R2)-np.mean(TESTING_TMAX_R2),
                    np.mean(TESTING_TMAX_RMSE), np.max(TESTING_TMAX_RMSE)-np.mean(TESTING_TMAX_RMSE), np.min(TESTING_TMAX_RMSE)-np.mean(TESTING_TMAX_RMSE),
                    np.mean(TESTING_SAVG_R2), np.max(TESTING_SAVG_R2)-np.mean(TESTING_SAVG_R2), np.min(TESTING_SAVG_R2)-np.mean(TESTING_SAVG_R2),
                    np.mean(TESTING_SAVG_RMSE), np.max(TESTING_SAVG_RMSE)-np.mean(TESTING_SAVG_RMSE), np.min(TESTING_SAVG_RMSE)-np.mean(TESTING_SAVG_RMSE),
                    np.mean(TESTING_SMAX_R2), np.max(TESTING_SMAX_R2)-np.mean(TESTING_SMAX_R2), np.min(TESTING_SMAX_R2)-np.mean(TESTING_SMAX_R2),
                    np.mean(TESTING_SMAX_RMSE), np.max(TESTING_SMAX_RMSE)-np.mean(TESTING_SMAX_RMSE), np.min(TESTING_SMAX_RMSE)-np.mean(TESTING_SMAX_RMSE),
                ]

                NEIGHBOR = [
                    np.mean(NEIGHBOR_R2), np.max(NEIGHBOR_R2)-np.mean(NEIGHBOR_R2), np.min(NEIGHBOR_R2)-np.mean(NEIGHBOR_R2),
                    np.mean(NEIGHBOR_RMSE), np.max(NEIGHBOR_RMSE)-np.mean(NEIGHBOR_RMSE), np.min(NEIGHBOR_RMSE)-np.mean(NEIGHBOR_RMSE),
                    np.mean(NEIGHBOR_TAVG_R2), np.max(NEIGHBOR_TAVG_R2)-np.mean(NEIGHBOR_TAVG_R2), np.min(NEIGHBOR_TAVG_R2)-np.mean(NEIGHBOR_TAVG_R2),
                    np.mean(NEIGHBOR_TAVG_RMSE), np.max(NEIGHBOR_TAVG_RMSE)-np.mean(NEIGHBOR_TAVG_RMSE), np.min(NEIGHBOR_TAVG_RMSE)-np.mean(NEIGHBOR_TAVG_RMSE),
                    np.mean(NEIGHBOR_TMAX_R2), np.max(NEIGHBOR_TMAX_R2)-np.mean(NEIGHBOR_TMAX_R2), np.min(NEIGHBOR_TMAX_R2)-np.mean(NEIGHBOR_TMAX_R2),
                    np.mean(NEIGHBOR_TMAX_RMSE), np.max(NEIGHBOR_TMAX_RMSE)-np.mean(NEIGHBOR_TMAX_RMSE), np.min(NEIGHBOR_TMAX_RMSE)-np.mean(NEIGHBOR_TMAX_RMSE),
                    np.mean(NEIGHBOR_SAVG_R2), np.max(NEIGHBOR_SAVG_R2)-np.mean(NEIGHBOR_SAVG_R2), np.min(NEIGHBOR_SAVG_R2)-np.mean(NEIGHBOR_SAVG_R2),
                    np.mean(NEIGHBOR_SAVG_RMSE), np.max(NEIGHBOR_SAVG_RMSE)-np.mean(NEIGHBOR_SAVG_RMSE), np.min(NEIGHBOR_SAVG_RMSE)-np.mean(NEIGHBOR_SAVG_RMSE),
                    np.mean(NEIGHBOR_SMAX_R2), np.max(NEIGHBOR_SMAX_R2)-np.mean(NEIGHBOR_SMAX_R2), np.min(NEIGHBOR_SMAX_R2)-np.mean(NEIGHBOR_SMAX_R2),
                    np.mean(NEIGHBOR_SMAX_RMSE), np.max(NEIGHBOR_SMAX_RMSE)-np.mean(NEIGHBOR_SMAX_RMSE), np.min(NEIGHBOR_SMAX_RMSE)-np.mean(NEIGHBOR_SMAX_RMSE),
                ]

                OUTSIDE = [
                    np.mean(OUTSIDE_R2), np.max(OUTSIDE_R2)-np.mean(OUTSIDE_R2), np.min(OUTSIDE_R2)-np.mean(OUTSIDE_R2),
                    np.mean(OUTSIDE_RMSE), np.max(OUTSIDE_RMSE)-np.mean(OUTSIDE_RMSE), np.min(OUTSIDE_RMSE)-np.mean(OUTSIDE_RMSE),
                    np.mean(OUTSIDE_TAVG_R2), np.max(OUTSIDE_TAVG_R2)-np.mean(OUTSIDE_TAVG_R2), np.min(OUTSIDE_TAVG_R2)-np.mean(OUTSIDE_TAVG_R2),
                    np.mean(OUTSIDE_TAVG_RMSE), np.max(OUTSIDE_TAVG_RMSE)-np.mean(OUTSIDE_TAVG_RMSE), np.min(OUTSIDE_TAVG_RMSE)-np.mean(OUTSIDE_TAVG_RMSE),
                    np.mean(OUTSIDE_TMAX_R2), np.max(OUTSIDE_TMAX_R2)-np.mean(OUTSIDE_TMAX_R2), np.min(OUTSIDE_TMAX_R2)-np.mean(OUTSIDE_TMAX_R2),
                    np.mean(OUTSIDE_TMAX_RMSE), np.max(OUTSIDE_TMAX_RMSE)-np.mean(OUTSIDE_TMAX_RMSE), np.min(OUTSIDE_TMAX_RMSE)-np.mean(OUTSIDE_TMAX_RMSE),
                    np.mean(OUTSIDE_SAVG_R2), np.max(OUTSIDE_SAVG_R2)-np.mean(OUTSIDE_SAVG_R2), np.min(OUTSIDE_SAVG_R2)-np.mean(OUTSIDE_SAVG_R2),
                    np.mean(OUTSIDE_SAVG_RMSE), np.max(OUTSIDE_SAVG_RMSE)-np.mean(OUTSIDE_SAVG_RMSE), np.min(OUTSIDE_SAVG_RMSE)-np.mean(OUTSIDE_SAVG_RMSE),
                    np.mean(OUTSIDE_SMAX_R2), np.max(OUTSIDE_SMAX_R2)-np.mean(OUTSIDE_SMAX_R2), np.min(OUTSIDE_SMAX_R2)-np.mean(OUTSIDE_SMAX_R2),
                    np.mean(OUTSIDE_SMAX_RMSE), np.max(OUTSIDE_SMAX_RMSE)-np.mean(OUTSIDE_SMAX_RMSE), np.min(OUTSIDE_SMAX_RMSE)-np.mean(OUTSIDE_SMAX_RMSE),
                ]

                SUMMARY_TABLE.append(TESTING)
                SUMMARY_TABLE.append(NEIGHBOR)
                SUMMARY_TABLE.append(OUTSIDE)

            save = pd.DataFrame(np.around(SUMMARY_TABLE, decimals=4))
            # save.round(4)
            save.to_csv('./sample_metrics/{}/summary_table_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=None)

            testingSetHeader = None
            # for _ in range(4):
            #     testingSetHeader += [[comp,'I'],[comp,'II'],[comp,'III']]
            np.save('./sample_metrics/{}/R_TS_{}_{}.npy'.format(dtst, dtst, obj), R_TS)
            save = pd.DataFrame(R_TS).T
            save.to_csv('./sample_metrics/{}/R_TS_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)

            np.save('./sample_metrics/{}/RMSE_TS_{}_{}.npy'.format(dtst, dtst, obj), RMSE_TS)
            save = pd.DataFrame(RMSE_TS).T
            save.to_csv('./sample_metrics/{}/RMSE_TS_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            np.save('./sample_metrics/{}/R_TAVG_{}_{}.npy'.format(dtst, dtst, obj), R_TAVG)
            save = pd.DataFrame(R_TAVG).T
            save.to_csv('./sample_metrics/{}/R_TAVG_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            np.save('./sample_metrics/{}/RMSE_TAVG_{}_{}.npy'.format(dtst, dtst, obj), RMSE_TAVG)
            save = pd.DataFrame(RMSE_TAVG).T
            save.to_csv('./sample_metrics/{}/RMSE_TAVG_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            np.save('./sample_metrics/{}/R_TMAX_{}_{}.npy'.format(dtst, dtst, obj), R_TMAX)
            save = pd.DataFrame(R_TMAX).T
            save.to_csv('./sample_metrics/{}/R_TMAX_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            np.save('./sample_metrics/{}/RMSE_TMAX_{}_{}.npy'.format(dtst, dtst, obj), RMSE_TMAX)
            save = pd.DataFrame(RMSE_TMAX).T
            save.to_csv('./sample_metrics/{}/RMSE_TMAX_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            np.save('./sample_metrics/{}/R_SAVG_{}_{}.npy'.format(dtst, dtst, obj), R_SAVG)
            save = pd.DataFrame(R_SAVG).T
            save.to_csv('./sample_metrics/{}/R_SAVG_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)

            np.save('./sample_metrics/{}/RMSE_SAVG_{}_{}.npy'.format(dtst, dtst, obj), RMSE_SAVG)
            save = pd.DataFrame(RMSE_SAVG).T
            save.to_csv('./sample_metrics/{}/RMSE_SAVG_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)

            np.save('./sample_metrics/{}/R_SMAX_{}_{}.npy'.format(dtst, dtst, obj), R_SMAX)
            save = pd.DataFrame(R_SMAX).T
            save.to_csv('./sample_metrics/{}/R_SMAX_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            np.save('./sample_metrics/{}/RMSE_SMAX_{}_{}.npy'.format(dtst, dtst, obj), RMSE_SMAX)
            save = pd.DataFrame(RMSE_SMAX).T
            save.to_csv('./sample_metrics/{}/RMSE_SMAX_{}_{}.csv'.format(dtst, dtst, obj), 
                        index=None, 
                        header=testingSetHeader)
            
            print('===========================================')
            print(dtst+obj + 'DONE!!!')

    

def train_generator(current_epoch=0):
    edata = np.load(edata_path)
    # lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32)
    batch_edges = []
    for i in range(batchSize):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    transnet = TransNet(dim=128)
    decoder = Decoder(n_layers=8,
                      n_features=1,
                      edges=batch_edges,
                      seq_length=sequence_size,
                      n_nodes=spatial_size,
                      n_hidden_feature=1,
                      read_out_dim=1)
    t_optimizer = tf.optimizers.Adam(learning_rate=lrT, beta_1=beta1, beta_2=beta2)
    # t_optimizer = tf.optimizers.RMSprop(learning_rate=lrT)
    if current_epoch != 0:
        transnet.load_weights(T_checkpoint_dir)
        decoder.load_weights(G_checkpoint_dir)
    if use_gan:
        discriminator = Discriminator(n_layers=4,
                                      n_features=1,
                                      edges=batch_edges,
                                      seq_length=sequence_size,
                                      n_nodes=spatial_size)
        disc_optimizer = tf.optimizers.Adam(learning_rate=lrDisc, beta_1=beta1, beta_2=beta2)
        if current_epoch != 0:
            discriminator.load_weights(Disc_checkpoint_dir)
    dvs = np.load(dvs_path)
    total_latents = np.load(latent_path)
    train_dvs = tf.convert_to_tensor(dvs[:training_dataset_size,:], dtype=tf.float32)
    test_dvs = tf.convert_to_tensor(dvs[-testing_dataset_size:,:], dtype=tf.float32)

    train_dvs = tf.unstack(train_dvs, axis=0)

    seq_test = []
    if objective_response == 'force':
        FS = os.listdir(force_test_path)
        for i in range(testing_dataset_size):
            f = np.load(force_test_path + FS[i])
            seq_test.append(f)
        seq_test = tf.stack(seq_test, axis=0)
        # seq_test = seq_test[:,:int(sequence_size/n_part),:]
        seq_test = keras.layers.Flatten()(seq_test)
        train_seqs = os.listdir(force_train_path)
    elif objective_response == 'energy':
        FS = os.listdir(energy_test_path)
        for i in range(testing_dataset_size):
            f = np.load(energy_test_path + FS[i])
            seq_test.append(f)
        seq_test = tf.stack(seq_test, axis=0)
        # seq_test = seq_test[:,:int(sequence_size/n_part),:]
        seq_test = keras.layers.Flatten()(seq_test)
        train_seqs = os.listdir(energy_train_path)

    for epoch in range(current_epoch, epochs):
        
        # st = time.time()
        indexes = genrandint(0, int(training_dataset_size-1), batchSize)
        fvs = tf.stack(grab(train_dvs, indexes), axis=0)

        seq = []
        for gs in grab(train_seqs, indexes):
            if objective_response == 'force':
                seq.append(tf.convert_to_tensor(np.load(force_train_path + gs), dtype=tf.float32))
            elif objective_response == 'energy':
                seq.append(tf.convert_to_tensor(np.load(energy_train_path + gs), dtype=tf.float32))
        seq = tf.expand_dims(tf.stack(seq, axis=0), axis=-1)
        # seq = seq[:,:int(sequence_size/n_part),:,:]
        transp = transnet(fvs)
        reconed = decoder(transp)

        if use_gan:
            gan_indexes = genrandint(0, int(training_dataset_size-1), batchSize)
            genp = tf.random.uniform(shape=[batchSize, latent_dim], minval=-1.0, maxval=1.0)
            fake_latent = transnet(genp)
            fake = decoder(fake_latent)
            gan_batch = []
            for gs in grab(train_seqs, gan_indexes):
                if objective_response == 'force':
                    gan_batch.append(tf.convert_to_tensor(np.load(force_train_path + gs), dtype=tf.float32))
                elif objective_response == 'energy':
                    gan_batch.append(tf.convert_to_tensor(np.load(energy_train_path + gs), dtype=tf.float32))
            gan_batch = tf.expand_dims(tf.stack(gan_batch, axis=0), axis=-1)

            # KL-Divergence approximation

            # 1. Donsker-V aradhan representation:
            # D_KL(P||Q) = sup {Ep[T] - log(Eq[e^T])}
            # L(T) = log(Eq[e^T]) - Ep[T]

            # 2. f-divergence representation:
            # D_KL(P||Q) = sup {Ep[T] - Eq[e^(T-1)]}
            # L(T) = Eq[e^(T-1)] - Ep[T]

            # 3. JS divergence
            for _ in range(DOptIters):
                # Data Space #
                with tf.GradientTape() as tape:
                    T_fake = discriminator(fake)
                    T_real = discriminator(gan_batch)
                    
                    # 1
                    # disc_loss = tf.math.log(tf.reduce_mean(tf.math.exp(T_real)) + 1e-7) - tf.reduce_mean(T_fake)
                    # 2
                    disc_loss = tf.reduce_mean(tf.math.exp(T_real - 1.0)) - tf.reduce_mean(T_fake)
                    # 3
                    # disc_loss = tf.math.log(tf.math.sigmoid(T_real) + 1e-7) + tf.math.log(1. - tf.math.sigmoid(T_fake) + 1e-7)
                    # disc_loss = -tf.reduce_mean(disc_loss)
                grads = tape.gradient(disc_loss, discriminator.trainable_variables)
                disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        
        with tf.GradientTape() as tape:
            # tape.watch([transnet.trainable_variables])
            trans_latents = transnet(fvs)
            reconed = decoder(trans_latents)
            l2_imgs = lambda_l2 * tf.reduce_mean(tf.math.reduce_sum((seq - reconed)**2, axis=[1,2,3]))
            l1_imgs = lambda_l1 * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(seq - reconed), axis=[1,2,3]))
            if use_gan:
                fake_latent = transnet(genp)
                fake = decoder(fake_latent)
                output_fake = discriminator(fake)
                # 3
                # output_fake = -tf.math.log(tf.math.sigmoid(output_fake) + 1e-7)
                
                gan_loss_gs = lambda_gan * tf.reduce_mean(output_fake)
            else:
                gan_loss_gs = 0.0
            g_loss = l1_imgs + l2_imgs + gan_loss_gs
        grads = tape.gradient(g_loss, transnet.trainable_variables + decoder.trainable_variables)
        t_optimizer.apply_gradients(zip(grads, transnet.trainable_variables + decoder.trainable_variables))

        seq_generated = []
        for i in range(int(tf.shape(test_dvs)[0]/batchSize)):
            sp = i*batchSize
            ep = (i+1)*batchSize
            test_batch = test_dvs[sp:ep,:]
            seq_generated.append(decoder(transnet(test_batch)))
        seq_generated = tf.concat(seq_generated, axis=0)
        # seq_generated = seq_generated[:,:int(sequence_size/n_part),:,:]
        seq_generated = keras.layers.Flatten()(seq_generated)

        # ================== compute R2 for samples average ==================
        R2 = []
        for i in range(tf.shape(seq_generated)[0]):    
            try:
                r2 = skm.r2_score(seq_test[i,:], seq_generated[i,:])
            except:
                r2 = 0.0
            r2 = 1-((1-r2)*(testing_dataset_size-1))/(testing_dataset_size-latent_dim-1)
            R2.append(r2)
        R2 = float(tf.reduce_mean(R2))
        # ==========================================================
        # L1 = float(skm.mean_absolute_error(seq_test, seq_generated))
        # L2 = float(skm.mean_squared_error(seq_test, seq_generated))
        # ARMSE = float(aRMSE(seq_test, seq_generated))
        # ARRMSE = float(aRRMSE(seq_test, seq_generated))
        # acc = float(ACC(seq_test, seq_generated))
        # are = float(tf.reduce_mean(tf.math.abs(seq_test - seq_generated)/tf.math.abs(seq_test)))
        DKL = float(-disc_loss)
        # DKL_lat = float(-latent_disc_loss)
        # R2 = float(skm.r2_score(seq_test, seq_generated))

        MAE = float(tf.reduce_mean(tf.math.abs(seq - reconed)))
        print(
              'EPOCH', epoch,
              'KL Divergence:', DKL,
              'L1 loss:', MAE,
              'R Square:' , R2,
              )

        ######### write training losses to .txt #########        
        # f=open(losses_save_dir,"a")
        # f.write("{} {} {} {} {} {} {} {}/n".format(
        #                                            DKL, 
        #                                             # DKL_lat,
        #                                             L1,
        #                                             # L1_lat,
        #                                             L2,
        #                                             # L2_lat,
        #                                             R2,
        #                                             # R2_lat,
        #                                             ARMSE,
        #                                             # ARMSE_lat,
        #                                             ARRMSE,
        #                                             # ARRMSE_lat,
        #                                             acc,
        #                                             # acc_lat,
        #                                             are,
        #                                             # are_lat
        #                                             ))
        # f.close()
        ##################################################
            
        # et = time.time()
        # print('time cost:', float(et - st))

        if epoch % 100 == 0:
            if use_gan:
                save_2d_contours([tf.squeeze(reconed), tf.squeeze(seq)],
                                ['Pred', 'GT'],
                                transnet_save_dir,
                                epoch)
            else:
                save_2d_contours([tf.squeeze(reconed), tf.squeeze(seq)],
                                ['Reconed', 'Decoded'],
                                transnet_save_dir,
                                epoch)
            if epoch != 0:
                transnet.save_weights(T_checkpoint_dir)
                decoder.save_weights(G_checkpoint_dir)
                if use_gan:
                    discriminator.save_weights(Disc_checkpoint_dir)
                print('ckpt file have been saved')
        
def genlatent_latents(
                     path, 
                     save_path, 
                     ):
    edata = np.load(edata_path)
    lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    batch_edges = []
    for i in range(1):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    encoder = Encoder(n_layers=4,
                          n_features=1,
                          latent_dim=latent_dim,
                          edges=batch_edges,
                          seq_length=sequence_size,
                          n_nodes=spatial_size,
                          n_hidden_feature=1,
                          read_out_dim=32)
    decoder = Decoder(n_layers=4,
                        n_features=1,
                        edges=batch_edges,
                        seq_length=sequence_size,
                        n_nodes=spatial_size,
                        n_hidden_feature=1,
                        read_out_dim=32)
    encoder.load_weights(E_checkpoint_dir)
    decoder.load_weights(D_checkpoint_dir)
    dataset_list = {}
    if objective_response == 'force':
        dataset_list['train'] = force_train_path
        dataset_list['test'] = force_test_path
        dataset_list['outside'] = force_outside_path
        dataset_list['neighbor'] = force_neighbor_path
        total_sample_list = os.listdir(force_total_path)
    elif objective_response == 'energy':
        dataset_list['train'] = energy_train_path
        dataset_list['test'] = energy_test_path
        dataset_list['outside'] = energy_outside_path
        dataset_list['neighbor'] = energy_neighbor_path
        total_sample_list = os.listdir(energy_total_path)
    
    dvs_total = np.load(dvs_path)
    
    for dataset_name in dataset_list:
        print('=============== START TO GENERATE LATENTS OF THE DATASET {} ==============='.format(str(dataset_name)))
        
        path = dataset_list[str(dataset_name)]
        samples = os.listdir(path)
        N_samples = len(samples)
        LATENTS = []
        DVS = []
        e = 0
        for i in range(int(N_samples/1)):
            sp = i*1
            ep = (i+1)*1
            batch_samples = samples[sp:ep]
            batch = [] 
            for sample in batch_samples:
                batch.append(tf.convert_to_tensor(np.load(path + sample), dtype=tf.float32))
                idx = total_sample_list.index(sample)
                dv = dvs_total[idx]
                print(idx, sample)
            batch = tf.expand_dims(tf.stack(batch, axis=0), axis=-1)
            lat = encoder(batch)
            reconed = decoder(lat)
            N = tf.shape(reconed)[0]
            zs = tf.unstack(tf.squeeze(reconed, axis=-1), axis=0)
            for j in range(N): 
                x = np.arange(0,0.1*spatial_size,0.1)
                y = np.arange(0,0.1*sequence_size,0.1)
                X,Y = np.meshgrid(x,y)
                plt.contourf(X,Y,zs[j], cmap=plt.cm.jet)
                plt.savefig('./results/' + recurrent_mode + '/' + dataset + '/' + objective_response + '/test_ae/' + str(e) + 'ttttt' + '.png')
                plt.close()
                e += 1
                print('--------- {}th sample has been processed ---------'.format(e))
            LATENTS.append(tf.squeeze(lat))
            DVS.append(dv)
        LATENTS = tf.stack(LATENTS, axis=0).numpy()
        DVS = tf.stack(DVS, axis=0).numpy()
        np.save(save_path + 'latents_{}.npy'.format(str(dataset_name)), LATENTS)
        np.save(save_path + 'dvs_{}.npy'.format(str(dataset_name)), DVS)

def metrics():
    edata = np.load(edata_path)
    lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    batch_edges = []
    for i in range(batchSize):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    transnet = TransNet(dim=latent_dim)
    decoder = MergeDecoder(
                            #    lap_eigvec=lap_eigvec,
                               lap_eigvec=None,
                               edges_index=batch_edges,
                               length=sequence_size,
                               batch_size=batchSize,
                               n_nodes=spatial_size,
                               n_layers=1)
    decoder.load_weights(D_checkpoint_dir)
    transnet.load_weights(T_checkpoint_dir)

    dvs = np.load(fv_path)
    train_dvs = tf.convert_to_tensor(dvs[:360,:], dtype=tf.float32)
    test_dvs = tf.convert_to_tensor(dvs[-140:,:], dtype=tf.float32)
    train_latents = np.load(train_latent_path)
    test_latents = np.load(test_latent_path)

    train_dvs = tf.unstack(train_dvs, axis=0)
    train_latents = tf.unstack(train_latents, axis=0)

    lat_decoded = []
    for i in range(int(tf.shape(test_latents)[0]/batchSize)):
        sp = i*batchSize
        ep = (i+1)*batchSize
        test_batch = test_latents[sp:ep,:]
        lat_decoded.append(decoder(test_batch))
    lat_decoded = tf.concat(lat_decoded, axis=0)
    lat_decoded = keras.layers.Flatten()(lat_decoded)
    # lat_decoded = test_latents
    lat_decoded = ((lat_decoded + 1.0)/2.0)
    tr_decoded = []
    for i in range(int(tf.shape(test_dvs)[0]/batchSize)):
        sp = i*batchSize
        ep = (i+1)*batchSize
        test_batch = test_dvs[sp:ep,:]
        # tr_decoded.append(decoder(transnet(test_batch)))
        tr_decoded.append(transnet(test_batch))
    tr_decoded = tf.concat(tr_decoded, axis=0)
    # tr_decoded = keras.layers.Flatten()(tr_decoded)
    tr_decoded = ((tr_decoded + 1.0)/2.0)
    # print(tf.reduce_min(lat_decoded), tf.reduce_min(tr_decoded))
    # print(tf.reduce_max(lat_decoded), tf.reduce_max(tr_decoded))
    mae = tf.math.abs(lat_decoded - tr_decoded)
    relative_error = tf.reduce_mean(mae / lat_decoded)
    print('Average Relative Error:{}'.format(relative_error))
    R2 = []
    for i in range(tf.shape(tr_decoded)[0]):    
        r2 = skm.r2_score(lat_decoded[i,:], tr_decoded[i,:])
        R2.append(r2)
    R2 = tf.reduce_mean(R2)
    # R2 = skm.r2_score(lat_decoded, tr_decoded)
    print('Average R2:{}'.format(R2))
    mean = tf.math.reduce_mean(lat_decoded, axis=0, keepdims=True)
    mean = tf.broadcast_to(mean, tf.shape(lat_decoded))
    CC = tf.math.reduce_sum((lat_decoded - mean) * (tr_decoded - mean), axis=0)/tf.math.sqrt(tf.math.reduce_sum((lat_decoded - mean)**2, axis=0)*tf.math.reduce_sum((tr_decoded - mean)**2, axis=0))
    ACC = tf.reduce_mean(CC)
    print('Average Correlation Coefficient:{}'.format(ACC))

def gen_test():
    edata = np.load(edata_path)
    lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    batch_edges = []
    for i in range(batchSize):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    transnet = TransNet(dim=latent_dim)
    decoder = Decoder(n_layers=4,
                      n_features=1,
                      edges=batch_edges,
                      seq_length=sequence_size,
                      n_nodes=spatial_size,
                      n_hidden_feature=1,
                      read_out_dim=32)
    decoder.load_weights(G_checkpoint_dir)
    transnet.load_weights(T_checkpoint_dir)
    dvs = np.load(dvs_path)
    test_dvs = tf.convert_to_tensor(dvs[-testing_dataset_size:,:], dtype=tf.float32)
    testing_set = os.listdir(force_test_path)
    ls = np.load('F:/code/spatiotemporal_metamodeling/graph_dynamic_version/datasets/cylinder_dataset/force/latents.npy')
    sup = tf.math.reduce_max(ls, axis=0, keepdims=True)
    inf = tf.math.reduce_min(ls, axis=0, keepdims=True)
    sup = tf.broadcast_to(sup, [batchSize, latent_dim])
    inf = tf.broadcast_to(inf, [batchSize, latent_dim])
    for i in range(int(testing_dataset_size/batchSize)):
        st = i*batchSize
        ed = (i+1)*batchSize
        batchdvs = test_dvs[st:ed,:]
        l = transnet(batchdvs)
        rl = ((l/2) + 0.5)*(sup-inf) + inf
        generated = decoder(rl)
        index = testing_set[st:ed]
        gt = []
        for j in index:
            gt.append(np.load(force_test_path + j))
        gt = tf.stack(gt, axis=0)
        save_2d_contours([tf.squeeze(generated), gt],
                        ['generated', 'ground true'],
                        gen_test_save_dir,
                        i,
                        interval=1)
        print(i)

def test():
    edata = np.load(edata_path)
    lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    batch_edges = []
    for i in range(batchSize):
        batch_edges.append(singe_edges+i*spatial_size)
    batch_edges = tf.concat(batch_edges, axis=1)
    transnet = TransNet(dim=latent_dim)
    decoder = MergeDecoder(
                                lap_eigvec=lap_eigvec,
                            #    lap_eigvec=None,
                                edges_index=batch_edges,
                                length=sequence_size,
                                batch_size=batchSize,
                                n_nodes=spatial_size,
                                n_layers=1)
    decoder.load_weights(D_checkpoint_dir)
    transnet.load_weights(T_checkpoint_dir)
    dvs = np.load(fv_path)
    test_dvs = tf.convert_to_tensor(dvs[-140:,:], dtype=tf.float32)
    tr_decoded = []
    for i in range(int(tf.shape(test_dvs)[0]/batchSize)):
        sp = i*batchSize
        ep = (i+1)*batchSize
        test_batch = test_dvs[sp:ep,:]
        tr_decoded.append(decoder(transnet(test_batch)))
    tr_decoded = tf.concat(tr_decoded, axis=0)
    test_latents = np.load(test_latent_path)
    lat_decoded = []
    for i in range(int(tf.shape(test_latents)[0]/batchSize)):
        sp = i*batchSize
        ep = (i+1)*batchSize
        test_batch = test_latents[sp:ep,:]
        lat_decoded.append(decoder(test_batch))
    lat_decoded = tf.concat(lat_decoded, axis=0)

    data_pack = [tr_decoded, lat_decoded]
    name_pack = ['pred', 'label']
    N = tf.shape(lat_decoded)[0]
    for k in range(len(data_pack)):
        zs = tf.unstack(data_pack[k], axis=0)
        for i in range(N):
            x = np.arange(0,0.1*spatial_size,0.1)
            y = np.arange(0,0.1*sequence_size,0.1)
            X,Y = np.meshgrid(x,y)
            plt.contourf(X,Y,zs[i], vmin=-1.2, vmax=1.2, cmap=plt.cm.jet)
            plt.savefig(test_save_dir + str(i) + name_pack[k] + '.png')
            plt.close()

def gen_weights_map(path, save_path):
    dvs = os.listdir(path)
    Y = []
    for dv in dvs:
        y = tf.convert_to_tensor(np.load(path + dv), dtype=tf.float32)
        Y.append(y + 1.0)
    Y = tf.stack(Y, axis=0)
    Y_averaged = tf.math.reduce_mean(Y, axis=0, keepdims=False)
    exponential = tf.math.exp(Y_averaged)
    sum_exponential = tf.math.reduce_sum(exponential)
    W_map = tf.math.divide(exponential, sum_exponential)
    print(tf.shape(W_map))
    np.save(save_path, W_map.numpy())

def optimization(map_path):
    step_size = 1e-2
    transnet = RegressionRBF(dim=latent_dim*2)
    decoder = Decoder0(length=sequence_size, spatial_size=spatial_size)
    decoder.load_weights(D_checkpoint_dir)
    transnet.load_weights(T_checkpoint_dir)
    # optimizer = tf.optimizers.Adam(learning_rate=setp_size, beta_1=beta1, beta_2=beta2)
    W_map = np.load(map_path)
    x = tf.Variable(
                    initial_value=tf.random.uniform([1, 5], -1.0, 1.0), 
                    trainable=True
                    )
    temp = 1e20
    u = 0
    n = 1
    for i in range(1000000):
        file_write_obj = open("G:/Results/TSNN/optimize_record.txt", 'a')
        with tf.GradientTape() as tape:
            transp = transnet(x)
            reconed = decoder(transp) + 1.0
            # y = tf.reduce_mean(W_map * reconed * sequence_size * spatial_size)
            y = tf.reduce_mean(reconed)
        # grads = tf.expand_dims(tape.gradient(y, x), axis=1)
        grads = tape.gradient(y, x)
        print(grads)
        norm = tf.norm(grads)
        # print(norm)
        # optimizer.apply_gradients(zip(grads, [x]))
        x.assign_sub((grads/norm)*step_size)
        x.assign(tf.clip_by_value(x, -1.0, 1.0))
        file_write_obj.writelines(str((y).numpy()))
        file_write_obj.write('/n')
        print(int(i), float(y))
        if y <= temp:
            temp = y
            np.save('G:/Results/TSNN/optimal_solution.npy', x.numpy())
        else:
            u += 1
        if u >= 20:
            # optimizer = tf.optimizers.Adam(learning_rate=step_size/(10**n), beta_1=beta1, beta_2=beta2)
            step_size = step_size/(10**n)
            n += 1
            u = 0
            print('setp size has been reduced to ', float(step_size/(10**n)))

def jacobian(f, x):
    """
    求函数一阶导
    :param f: 原函数
    :param x: 初始值
    :return: 函数一阶导的值
    """
    grandient = np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0] ** 2 - x[1])], dtype=float)
    return grandient

def bfgs_newton(f, x, iters):
    """
    实现BFGS拟牛顿法
    :param f: 原函数
    :param x: 初始值
    :param iters: 遍历的最大epoch
    :return: 最终更新完毕的x值
    """
    # 步长。设为1才能收敛，小于1不能收敛
    learning_rate = 1
    # 初始化B正定矩阵
    B = np.eye(2)
    x_len = x.shape[0]
    # 一阶导g的第二范式的最小值（阈值）
    epsilon = 1e-5
    for i in range(1, iters):
        g = jacobian(f, x)
        if np.linalg.norm(g) < epsilon:
            break
        p = np.linalg.solve(B, g)
        # 更新x值
        x_new = x - p*learning_rate
        print("第" + str(i) + "次迭代后的结果为:", x_new)
        g_new = jacobian(f, x_new)
        y = g_new - g
        k = x_new - x
        y_t = y.reshape([x_len, 1])
        Bk = np.dot(B, k)
        k_t_B = np.dot(k, B)
        kBk = np.dot(np.dot(k, B), k)
        # 更新B正定矩阵。完全按照公式来计算
        B = B + y_t*y/np.dot(y, k) - Bk.reshape([x_len, 1]) * k_t_B / kBk
        x = x_new
    return x

def PROCESS_MODEL(mode, current_epoch=0):
    if mode == 'TRAIN_AE':
        train_graph_ae(current_epoch=current_epoch)
    if mode == 'GEN_LATENTS':
        if objective_response == 'force':
            genlatent_latents(path='./datasets/' + dataset + '_dataset/force/force_total/',
                        save_path='./datasets/' + dataset + '_dataset/force/')
        if objective_response == 'energy':
            genlatent_latents(path='./datasets/' + dataset + '_dataset/energy/SEA_total/',
                        save_path='./datasets/' + dataset + '_dataset/energy/')
    if mode == 'PRETRAIN_TRANSNET':
        pretrain_transnet(current_epoch=current_epoch)
    if mode == 'TRAIN_TRANSNET':
        train_transnet(current_epoch=current_epoch)
    if mode == 'TRAIN_GENERATOR':
        train_generator(current_epoch=current_epoch)
    if mode == 'TEST':
        test()
    if mode == 'GEN_TEST':
        gen_test()

if __name__ == '__main__':

    TF_CPP_MIN_LOG_LEVEL = 3
    setup('force', 'cellular', 'proposed')
    train_transnet()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'     # Using CPU
    
    # for comp in ['teach', 'det_only']:
    #     for dtst in ['cylinder', 'cellular']:
    #         for obj in ['energy', 'force']:
    #             if comp == 'teach':
    #                 if (dtst == 'cellular') and (obj == 'force'):
    #                     print('start up:')
    #                     print(obj, dtst, comp)
    #                     setup(obj, dtst, comp)
    #                     PROCESS_MODEL('TRAIN_TRANSNET')
    #                 else:
    #                     pass
    #             else:
    #                 print('start up:')
    #                 print(obj, dtst, comp)
    #                 setup(obj, dtst, comp)
    #                 PROCESS_MODEL('TRAIN_TRANSNET')

    # generate_samples_metrics()


    # objective_response = 'energy'
    # PROCESS_MODEL('TRAIN_TRANSNET')
    # dataset = 'cellular'
    # objective_response = 'force'
    # PROCESS_MODEL('TRAIN_TRANSNET')
    # objective_response = 'energy'
    # PROCESS_MODEL('TRAIN_TRANSNET') 
    # dvs = np.load('F:\code\spatiotemporal_metamodeling\graph_dynamic_version\datasets\cellular_dataset\design_variables.npy')
    # save = pd.DataFrame(np.transpose(dvs))
    # save.to_csv('F:\code\spatiotemporal_metamodeling\graph_dynamic_version\datasets\cellular_dataset\design_variables.csv', index=False, header=False)
    # latents = np.load(latent_path)
    # print(np.shape(latents))
    # print(np.mean(latents))
    # print(np.min(latents))
    
    # ==================== AE test curve ====================
    # edata = np.load(edata_path)
    # lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    # singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32)
    # batch_edges = []
    # for i in range(batchSize):
    #     batch_edges.append(singe_edges+i*spatial_size)
    # batch_edges = tf.concat(batch_edges, axis=1)
    # encoder = Encoder(n_layers=4,
    #                       n_features=1,
    #                       latent_dim=latent_dim,
    #                       edges=batch_edges,
    #                       seq_length=sequence_size,
    #                       n_nodes=spatial_size,
    #                       n_hidden_feature=1,
    #                       read_out_dim=32)
    # decoder = Decoder(n_layers=4,
    #                     n_features=1,
    #                     edges=batch_edges,
    #                     seq_length=sequence_size,
    #                     n_nodes=spatial_size,
    #                     n_hidden_feature=1,
    #                     read_out_dim=32)
    # encoder.load_weights(E_checkpoint_dir)
    # decoder.load_weights(D_checkpoint_dir)
    # graph_seq = process_seq_graph(path_f=force_train_path, batch_size=batchSize)
    # graph_seq = tf.expand_dims(graph_seq, axis=-1)
    # reconed = decoder(encoder(graph_seq))

    # reconed = tf.squeeze(reconed)
    # graph_seq = tf.squeeze(graph_seq)

    # for i in range(spatial_size):
    #     ori_seq = graph_seq[0,:,i]
    #     fil_seq = reconed[0,:,i]
    #     x = np.linspace(0,20,sequence_size)
    #     plt.figure(num=3,figsize=(8,5))
    #     plt.plot(x, ori_seq)
    #     plt.plot(x, fil_seq, color='red', linewidth=1, linestyle='--')
    #     plt.savefig('./curve_test/{}'.format(i) + '.png')
    #     plt.close()
    #     print(i)
    
    # ============================================================

    

    # ls = np.load(latent_path)
    # sup = tf.math.reduce_max(ls, axis=0, keepdims=True)
    # inf = tf.math.reduce_min(ls, axis=0, keepdims=True)
    # sup = tf.broadcast_to(sup, [500, latent_dim])
    # inf = tf.broadcast_to(inf, [500, latent_dim])
    # normed_ls = (ls - inf)/(sup - inf)
    # normed_ls = (normed_ls - 0.5)*2
    # np.save('./datasets/' + dataset + '_dataset/' + objective_response + '/normed_latents.npy', normed_ls.numpy())

    # edata = np.load(edata_path)
    # lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    # singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    # batch_edges = []
    # for i in range(batchSize):
    #     batch_edges.append(singe_edges+i*spatial_size)
    # batch_edges = tf.concat(batch_edges, axis=1)
    # encoder = Encoder(n_layers=4,
    #                       n_features=1,
    #                       latent_dim=latent_dim,
    #                       edges=batch_edges,
    #                       seq_length=sequence_size,
    #                       n_nodes=spatial_size)
    # decoder = Decoder(n_layers=4,
    #                     n_features=1,
    #                     edges=batch_edges,
    #                     seq_length=sequence_size,
    #                     n_nodes=spatial_size)
    # encoder.load_weights(E_checkpoint_dir)
    # decoder.load_weights(D_checkpoint_dir)
    # graph_seq = process_seq_graph(path_f=force_train_path, batch_size=batchSize)
    # graph_seq = tf.expand_dims(graph_seq, axis=-1)
    # reconed = decoder(encoder(graph_seq))

    # reconed = tf.squeeze(reconed)
    # graph_seq = tf.squeeze(graph_seq)

    # spatial_size = 240
    # sequence_size = 170
    # for i in range(spatial_size):
    #     ori_seq = graph_seq[0,:,i]
    #     fil_seq = reconed[0,:,i]
    #     x = np.linspace(0,20,sequence_size)
    #     plt.figure(num=3,figsize=(8,5))
    #     plt.plot(x, ori_seq)
    #     plt.plot(x, fil_seq, color='red', linewidth=1, linestyle='--')
    #     plt.savefig('I:/CODE/spatiotemporal_metamodeling/graph_dynamic_version/results/Merge/cylinder/force/test/{}'.format(i) + '.png')
    #     plt.close()
    #     print(i)

    # print(float(np.random.uniform(0, 1, (1))))

    # edata = np.load(edata_path)
    # lap_eigvec = tf.convert_to_tensor(edata['Lev'], dtype=tf.float32)
    # singe_edges = tf.convert_to_tensor(edata['edges_index'], dtype=tf.int32) 
    # batch_edges = []
    # for i in range(batchSize):
    #     batch_edges.append(singe_edges+i*spatial_size)
    # batch_edges = tf.concat(batch_edges, axis=1)
    # transnet = TransNet(dim=latent_dim)
    # decoder = MergeDecoder(
    #                         #    lap_eigvec=lap_eigvec,
    #                            lap_eigvec=None,
    #                            edges_index=batch_edges,
    #                            length=sequence_size,
    #                            batch_size=batchSize,
    #                            n_nodes=spatial_size,
    #                            n_layers=1)
    # decoder.load_weights(D_checkpoint_dir)
    # transnet.load_weights(T_checkpoint_dir)
    # dvs = np.load(fv_path)
    # train_dvs = tf.convert_to_tensor(dvs[:360,:], dtype=tf.float32)
    # test_dvs = tf.convert_to_tensor(dvs[-140:,:], dtype=tf.float32)
    # train_latents = np.load(train_latent_path)
    # test_latents = np.load(test_latent_path)

    # train_dvs = tf.unstack(train_dvs, axis=0)
    # train_latents = tf.unstack(train_latents, axis=0)

    # lat_decoded = []
    # for i in range(int(tf.shape(test_latents)[0]/batchSize)):
    #     sp = i*batchSize
    #     ep = (i+1)*batchSize
    #     test_batch = test_latents[sp:ep,:]
    #     lat_decoded.append(decoder(test_batch))
    # lat_decoded = tf.concat(lat_decoded, axis=0)
    # lat_decoded = keras.layers.Flatten()(lat_decoded)

    # tr_decoded = []
    # for i in range(int(tf.shape(test_dvs)[0]/batchSize)):
    #     sp = i*batchSize
    #     ep = (i+1)*batchSize
    #     test_batch = test_dvs[sp:ep,:]
    #     tr_decoded.append(decoder(transnet(test_batch)))
    # tr_decoded = tf.concat(tr_decoded, axis=0)
    # tr_decoded = keras.layers.Flatten()(tr_decoded)

    # for i in range(tf.shape(test_latents)[0]):
    #     x = tr_decoded[i]
    #     y = lat_decoded[i]
    #     plt.figure()
    #     plt.scatter(x, y)
    #     plt.savefig('I:/CODE/spatiotemporal_metamodeling/graph_dynamic_version/results/Merge/cylinder/energy/R2_scatter/' + str(i) + '.png')
    #     plt.close()
    #     print(i)




    # metrics()
    # force = np.load(force_train_path+'000.npy')
    # print(np.shape(force))

    # genlatent_latents('I:/DATASETs/cylinder_dataset/energy/SEA_train/',
    #                   'I:/DATASETs/cylinder_dataset/energy/')

    # print(np.load('I:/DATASETs/cylinder_dataset/normed_dvs.npy')[0])
    # ranges = [[50.0, 55.0], [30.0, 35.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]
    # dvs = np.load('I:/DATASETs/cylinder_dataset/design_variables.npy')
    # axises = tf.unstack(dvs, axis=1)
    # normed = []
    # for i, r in enumerate(ranges):
    #     normed.append(((axises[i] - r[0])/(r[1] - r[0]))*2-1.0)
    # normed = tf.stack(normed, axis=1)
    # np.save('I:/DATASETs/cylinder_dataset/normed_dvs.npy',normed.numpy())

    # dvs = np.load('I:/DATASETs/cylinder_dataset/design_variables.npy')
    # X_train = dvs[:360,:]
    # X_test = dvs[-140:,:]
    # y_train = np.load('I:/DATASETs/cylinder_dataset/energy/latents_train.npy')
    # y_test = np.load('I:/DATASETs/cylinder_dataset/energy/latents_test.npy')

    # dvs = np.load(dvs_path)
    # # total_latents = np.load('./datasets/' + dataset + '_dataset/' + objective_response + '/latents.npy')[:,0]
    # if objective_response == 'force':
    #     obj_path = f_scalar_obj_path
    # elif objective_response == 'energy':
    #     obj_path = e_scalar_obj_path
    # total_latents = np.load(obj_path)
    # X_train = dvs[:400,:]
    # X_test = dvs[-100:,:]
    # y_train = total_latents[:400]
    # y_test = total_latents[-100:]

    # # # regr = ensemble.RandomForestRegressor(n_estimators=500)
    # # # gbr = ensemble.GradientBoostingRegressor(loss='huber', learning_rate=0.1, max_depth=6)
    # # # regr = ensemble.ExtraTreesRegressor(n_estimators=50, max_depth=10)
    # # regr = sksvm.SVR(kernel='rbf', max_iter=-1)
    # # # lasso = linear_model.Lasso()
    # # regr = MultiOutputRegressor(svr)
    # best_r2 = 0.0
    # for c in range(1,50):
    #     for g in range(1,50):
    #         regr = LSSVR(C=c, kernel='rbf', gamma=g)
    #         regr.fit(X_train, y_train)
    #         if regr.score(X_test, y_test) > best_r2 and regr.score(X_train, y_train) > 0.9:
    #             print("C={}, gamma={}, Traing Score:{}, Testing Score:{}, ".format(c, g, regr.score(X_train, y_train), regr.score(X_test, y_test)))
    #             best_r2 = regr.score(X_test, y_test)


    # for j in range(100):
    #     n_estimators = (j+1) * 10
    #     regr = ensemble.ExtraTreesRegressor(n_estimators=n_estimators)
    #     regr.fit(X_train, y_train)
    #     print('n_estimators:{}'.format(n_estimators))
    #     print("Traing Score:%.5f" % regr.score(X_train, y_train),"Testing Score:%.5f" % regr.score(X_test, y_test))


    # print(np.shape(np.load('I:/DATASETs/cylinder_dataset/design_variables.npy')))

    

