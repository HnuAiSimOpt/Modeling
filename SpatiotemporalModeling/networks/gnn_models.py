import tf_geometric as tfg
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional
import time
from networks.kernalize import RandomFourierFeatures
from networks.customed import SwitchNormalization, ConvSN2D



# def tf.nn.leaky_relu(x):
#     return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / tf.convert_to_tensor(np.pi, dtype=tf.float32)) * (x + 0.044715 * tf.math.pow(x, 3))))

'''
    BASIC MODULEs
'''

class ResUpBlock(keras.Model):
    def __init__(self, ch, strides):
        super(ResUpBlock, self).__init__()
        self.strides = strides
        if conduct_sn:
            self.conv1 = ConvSN2D(filters=ch, kernel_size=[1, 1], strides=1, padding='same')
            self.conv2 = ConvSN2D(filters=ch, kernel_size=[3, 3],strides=1, padding='same')
            self.convid = ConvSN2D(filters=ch, kernel_size=[1, 1],strides=1, padding='same')
        else:
            self.conv1 = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1], strides=1, padding='same')
            self.conv2 = keras.layers.Conv2D(filters=ch, kernel_size=[3, 3],strides=1, padding='same')
            self.convid = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=1, padding='same')
        self.sn1 = SwitchNormalization()
        self.sn2 = SwitchNormalization()
        self.sn4 = SwitchNormalization()
    def call(self, inputs, training=None):
        y = self.conv1(tf.nn.leaky_relu(self.sn1(inputs)))
        y = tf.keras.backend.resize_images(y, self.strides, self.strides, data_format='channels_last')
        y = self.conv2(tf.nn.leaky_relu(self.sn2(y)))

        identity = tf.keras.backend.resize_images(inputs, self.strides, self.strides, data_format='channels_last')
        identity = self.convid(tf.nn.leaky_relu(self.sn4(identity)))
        return y + identity

class ResDownBlock(keras.Model):
    def __init__(self, ch, strides):
        super(ResDownBlock, self).__init__()
        self.strides = strides
        if conduct_sn:
            self.conv1 = ConvSN2D(filters=ch, kernel_size=[3, 3],strides=1, padding='same')
            self.conv2 = ConvSN2D(filters=ch, kernel_size=[1, 1],strides=strides, padding='same')
            self.convid = ConvSN2D(filters=ch, kernel_size=[1, 1],strides=strides, padding='same')
        else:
            self.conv1 = keras.layers.Conv2D(filters=ch, kernel_size=[3, 3],strides=1, padding='same')
            self.conv2 = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=strides, padding='same')
            self.convid = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=strides, padding='same')
        self.sn1 = SwitchNormalization()
        self.sn2 = SwitchNormalization()
        self.sn4 = SwitchNormalization()
    def call(self, inputs, training=None):
        y = self.conv1(tf.nn.leaky_relu(self.sn1(inputs)))
        y = self.conv2(tf.nn.leaky_relu(self.sn2(y)))
        identity = self.convid(tf.nn.leaky_relu(self.sn4(inputs)))
        return y + identity

class DenseLayer(keras.Model):
    def __init__(self, u):
        super(DenseLayer, self).__init__()
        self.dense = keras.layers.Dense(u)
        self.sn = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        return tf.nn.leaky_relu(self.sn(self.dense(inputs)))

class Conv1DLayer(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, activation='non_linear'):
        super(Conv1DLayer, self).__init__()
        self.activation = activation
        self.layer = keras.layers.Conv1D(filters=filters, 
                                          kernel_size=kernel_size, 
                                          strides=strides, 
                                          padding=padding)
        self.norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        y1 = self.layer(inputs)
        if self.activation == 'non_linear':
            return tf.nn.leaky_relu(self.norm(y1))
        elif self.activation == 'linear':
            return self.norm(y1)

class ResConv2D(keras.Model):
    def __init__(self, ch, strides, activation='non_linear'):
        super(ResConv2D, self).__init__()
        self.strides = strides
        self.activation = activation
        self.conv1 = keras.layers.Conv2D(filters=ch, kernel_size=[3, 3],strides=1, padding='same')
        self.conv2 = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=strides, padding='same')
        self.convid = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=strides, padding='same')
        self.sn1 = keras.layers.BatchNormalization()
        self.sn2 = keras.layers.BatchNormalization()
        self.sn4 = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        y = self.conv1(tf.nn.leaky_relu(self.sn1(inputs)))
        y = self.conv2(tf.nn.leaky_relu(self.sn2(y)))
        identity = self.convid(tf.nn.leaky_relu(self.sn4(inputs)))
        if self.activation == 'linear':
            return y + identity
        elif self.activation == 'non_linear':
            return tf.nn.leaky_relu(y + identity)



class DenseBlock(keras.Model):
    def __init__(self, out_dim, activation='non_linear'):
        super(DenseBlock, self).__init__()
        self.activation = activation
        self.layer1 = keras.layers.Dense(out_dim)
        self.layer2 = keras.layers.Dense(out_dim)
        self.layer3 = keras.layers.Dense(out_dim)
        self.norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        y1 = self.layer1(inputs)
        y2 = self.layer2(y1)
        y3 = self.layer3(y1 + y2)
        if self.activation == 'non_linear':
            return tf.nn.leaky_relu(self.norm(y1 + y2 + y3))
        if self.activation == 'linear':
            return self.norm(y1 + y2 + y3)

class DenseConv1D(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, activation='non_linear'):
        super(DenseConv1D, self).__init__()
        self.activation = activation
        self.layer1 = keras.layers.Conv1D(filters=filters, 
                                          kernel_size=kernel_size, 
                                          strides=strides, 
                                          padding=padding)
        self.layer2 = keras.layers.Conv1D(filters=filters, 
                                          kernel_size=kernel_size, 
                                          strides=strides, 
                                          padding=padding)
        self.layer3 = keras.layers.Conv1D(filters=filters, 
                                          kernel_size=kernel_size, 
                                          strides=strides, 
                                          padding=padding)
        self.norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        y1 = self.layer1(inputs)
        y2 = self.layer2(y1)
        y3 = self.layer3(y1 + y2)
        if self.activation == 'non_linear':
            return tf.nn.leaky_relu(self.norm(y1 + y2 + y3))
        elif self.activation == 'linear':
            return self.norm(y1 + y2 + y3)
    
class ResRBFDenseBlock(keras.Model):
    def __init__(self, out_dim, activation='non_linear', use_norm=True, rbf_dim=1024):
        super(ResRBFDenseBlock, self).__init__()
        self.layer1 = RBFDense(out_dim=out_dim, activation=activation, use_norm=use_norm, rbf_dim=rbf_dim)
        self.layer2 = RBFDense(out_dim=out_dim, activation=activation, use_norm=use_norm, rbf_dim=rbf_dim)
        self.layer3 = RBFDense(out_dim=out_dim, activation=activation, use_norm=use_norm, rbf_dim=rbf_dim)

    def call(self, inputs, training=None):
        y1 = self.layer1(inputs)
        y2 = self.layer2(y1)
        y3 = self.layer3(y1 + y2)
        return y1 + y2 + y3

class GraphNorm(keras.layers.Layer):
    '''
    tensorflow 2.x implementation of Graph Normalization
    '''
    def __init__(self, batch_size, n_features, n_units, eps=1e-5, is_node=True):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps
        self.n_features = n_features
        self.is_node = is_node

        self.gamma = tf.Variable(tf.ones([self.n_features, n_units]))
        self.beta = tf.Variable(tf.ones([self.n_features, n_units]))

    def norm(self, x):

        mean = tf.math.reduce_mean(x, axis=0, keepdims=True)
        var = tf.math.reduce_std(x, axis=0, keepdims=True)
        x = (x - mean) / (var + self.eps)
        return x

    def call(self, x):

        x_list = tf.split(x, self.batch_size, axis=0)
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = tf.concat(norm_list, axis=0)
        return self.gamma * norm_x + self.beta

class GATLayer(keras.layers.Layer):
    def __init__(self, units, batch_size, n_features, activation='non_linear'):
        super(GATLayer, self).__init__()

        self.activation = activation
        self.gat = tfg.layers.GAT(units=units, 
                                  num_heads=1, 
                                  query_activation=tf.nn.leaky_relu, 
                                  key_activation=tf.nn.leaky_relu)
        self.graph_norm = GraphNorm(batch_size=batch_size, 
                            n_features=n_features,
                            n_units=units)
    
    def call(self, inputs, training=None):
        g, e = inputs[0], inputs[1]
        y = self.gat([g, e])
        y = self.graph_norm(y)
        if self.activation == 'linear':
            return y
        elif self.activation == 'non_linear':
            return tf.nn.leaky_relu(y)

class GNNBlock(keras.Model):
    def __init__(self, n_layers, units, batch_size, n_features, activation='non_linear'):
        super(GNNBlock, self).__init__()

        self.n_layers = n_layers
        self.activation = activation
        self.gnn_layers = []
        self.g_norm_layers = []
        for _ in range(n_layers):
            block = []
            # block.append(tfg.layers.GIN(mlp_model=DenseBlock(units),
            #                      train_eps=True))
            # block.append(tfg.layers.GIN(mlp_model=DenseBlock(units),
            #                      train_eps=True))
            
            # block.append(tfg.layers.GIN(mlp_model=keras.layers.Dense(units),
            #                      train_eps=True))
            # block.append(tfg.layers.GIN(mlp_model=keras.layers.Dense(units),
            #                      train_eps=True))
            block.append(tfg.layers.GAT(units=units, 
                                        num_heads=1, 
                                        query_activation=tf.nn.leaky_relu, 
                                        key_activation=tf.nn.leaky_relu))
            block.append(tfg.layers.GAT(units=units, 
                                        num_heads=1, 
                                        query_activation=tf.nn.leaky_relu, 
                                        key_activation=tf.nn.leaky_relu))
            self.gnn_layers.append(block)
        for _ in range(n_layers):
            self.g_norm_layers.append(GraphNorm(batch_size=batch_size, 
                            n_features=n_features,
                            n_units=units))

    def call(self, inputs, training=None):

        g, e = inputs[0], inputs[1]
        for i in range(self.n_layers-1):
            p1 = self.gnn_layers[i][0]([g, e])
            p2 = self.gnn_layers[i][1]([g, e])
            g = tf.nn.leaky_relu(self.g_norm_layers[i](p1 + p2))
        
        p1 = self.gnn_layers[-1][0]([g, e])
        p2 = self.gnn_layers[-1][1]([g, e])
        g = self.g_norm_layers[-1](p1 + p2)
        if self.activation == 'non_linear':
            return tf.nn.leaky_relu(g)
        elif self.activation == 'linear':
            return g

class IntegralGraph(keras.layers.Layer):
    def __init__(self, lap_eigvec, batch_size, n_nodes, latent_dim, use_lapembd = False):
        super(IntegralGraph, self).__init__()
        self.batch_size = batch_size
        self.lap_eigvec = lap_eigvec
        self.use_lapembd = use_lapembd
        if lap_eigvec is not None:
            self.lap_embd = DenseBlock(1)
        # self.w_graph = tfg.layers.GIN(mlp_model=keras.layers.Dense(latent_dim),
        #                                      train_eps=True)
        # self.w_graph_norm = keras.layers.BatchNormalization()
        self.w_graph = GNNBlock(1, latent_dim, batch_size, batch_size*n_nodes)                     
        # self.integral_graph = tfg.layers.GIN(mlp_model=keras.layers.Dense(1),
        #                                      train_eps=True)
        # self.integral_graph_norm = keras.layers.BatchNormalization()
        self.integral_graph = GNNBlock(1, 1, batch_size, batch_size*n_nodes)
        self.outlayer = DenseBlock(latent_dim)

    def call(self, inputs, training=None):
        seq_integrated, e_index = inputs[0], inputs[1]
        # calculating the weights for integrating graph
        w_graph = self.w_graph([seq_integrated, e_index])
        w_graph = tf.stack(tf.split(w_graph, self.batch_size, 0), axis=0)
        # w_graph = tf.nn.leaky_relu(self.w_graph_norm(w_graph))
        w_graph = tf.nn.softmax(w_graph, axis=1)
        seq_integrated = self.integral_graph([seq_integrated, e_index])
        seq_integrated = tf.stack(tf.split(seq_integrated, self.batch_size, 0), axis=0)
        # seq_integrated = tf.nn.leaky_relu(self.integral_graph_norm(seq_integrated))
        seq_integrated = tf.broadcast_to(seq_integrated, tf.shape(w_graph))
        if self.lap_eigvec is not None:
            lap = self.lap_embd(self.lap_eigvec)
            lap = tf.broadcast_to(lap, tf.shape(w_graph))
        else:
            lap = 0.
        # -->   w : [batch_size, n_nodes, latent_dim]
        # --> seq : [batch_size, n_nodes, latent_dim]
        integral = tf.reduce_sum(w_graph * (seq_integrated + lap), axis=1)
        return self.outlayer(integral)

class IntegralLayer(keras.layers.Layer):
    def __init__(self, lap_eigvec, batch_size, n_nodes, latent_dim, length, in_dim):
        super(IntegralLayer, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.in_dim = in_dim
        self.lap_eigvec = lap_eigvec
        self.lap_embd = keras.layers.Dense(1)
        self.w_seq = DenseConv1D(filters=1, 
                                kernel_size=batch_size*n_nodes, 
                                strides=1, 
                                padding='valid')
        self.w_graph = tfg.layers.GIN(mlp_model=keras.layers.Dense(latent_dim),
                                             train_eps=True)
        self.w_graph_norm = keras.layers.BatchNormalization()
        self.integral_graph = tfg.layers.GIN(mlp_model=keras.layers.Dense(1),
                                             train_eps=True)
        self.integral_graph_norm = keras.layers.BatchNormalization()
        self.outlayer = keras.layers.Dense(latent_dim)

    def call(self, inputs, training=None):
        seq_graph, e_index = inputs[0], inputs[1]
        seq_graph = self.positional_encoding(seq_graph)
        # calculating the weights for integration along sequence axis
        w_seq = tf.nn.leaky_relu(self.w_seq(seq_graph))
        w_seq = tf.nn.softmax(w_seq, axis=0)
        w_seq = tf.broadcast_to(w_seq, tf.shape(seq_graph))
        seq_integrated = tf.reduce_sum(w_seq * seq_graph, axis=0)
        # calculating the weights for integrating graph
        w_graph = self.w_graph([seq_integrated, e_index])
        w_graph = tf.stack(tf.split(w_graph, self.batch_size, 0), axis=0)
        w_graph = tf.nn.leaky_relu(self.w_graph_norm(w_graph))
        w_graph = tf.nn.softmax(w_graph, axis=1)
        seq_integrated = self.integral_graph([seq_integrated, e_index])
        seq_integrated = tf.stack(tf.split(seq_integrated, self.batch_size, 0), axis=0)
        seq_integrated = tf.nn.leaky_relu(self.integral_graph_norm(seq_integrated))
        seq_integrated = tf.broadcast_to(seq_integrated, tf.shape(w_graph))
        lap = self.lap_embd(self.lap_eigvec)
        lap = tf.broadcast_to(lap, tf.shape(w_graph))
        # -->   w : [batch_size, n_nodes, latent_dim]
        # --> seq : [batch_size, n_nodes, latent_dim]
        integral = tf.reduce_sum(w_graph * (seq_integrated + lap), axis=1)
        return self.outlayer(integral)
    
    def positional_encoding(self, seq):
        shape = tf.shape(seq)
        pos = np.expand_dims(np.arange(0, self.length), axis=1)
        index = np.expand_dims(np.arange(0, self.in_dim), axis=0)

        pe = pos / np.power(10000, (index - index % 2) / np.float32(self.in_dim))

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=1)
        pe = tf.cast(tf.broadcast_to(pe, shape), dtype=tf.float32)
        seq += pe
        return seq

# class ConvEncoder(keras.Model):
#     def __init__(self,
#                  ch,
#                  n_layers,
#                  ):

class ShuntLayer(keras.layers.Layer):
    def __init__(self, 
                 lap_eigvec, 
                 batch_size, 
                 n_nodes, 
                 length, 
                 embedding_dim = 1):
        super(ShuntLayer, self).__init__()
        self.length = length
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.lap_eigvec = lap_eigvec
        if lap_eigvec is not None:
            self.embed_lap = DenseBlock(embedding_dim)
        self.to_grap_skeleton = DenseBlock(n_nodes)
        self.to_graph = Conv1DLayer(filters=embedding_dim, kernel_size=3, strides=1, padding='same')
        # self.to_graph_norm = keras.layers.BatchNormalization()
        self.gnn = GNNBlock(n_layers=1, 
                            units=embedding_dim, 
                            batch_size=batch_size, 
                            n_features=batch_size*n_nodes)
        # self.gnn = tfg.layers.GIN(mlp_model=keras.layers.Dense(embedding_dim),
        #                          train_eps=True)
        # self.gnn_norm = GraphNorm(batch_size=batch_size, n_features=batch_size*n_nodes, n_units=embedding_dim)
        self.qvk_conv = Conv1DLayer(filters=embedding_dim*3*length, kernel_size=3, strides=1, padding='same')
        # self.qvk_conv_norm = keras.layers.BatchNormalization()
        self.attention = keras.layers.Attention()
    def call(self, inputs, training=None):
        latent, edges_index = inputs[0], inputs[1]

        # expand to graph
        # --- [batch_size, latent_dim]
        skeleton = self.to_grap_skeleton(latent)
        skeleton = tf.expand_dims(skeleton, axis=-1)
        skeleton = self.to_graph(skeleton)
        if self.lap_eigvec is not None:
            embd_lap = self.embed_lap(self.lap_eigvec)
            embd_lap = tf.broadcast_to(tf.expand_dims(embd_lap, axis=0), tf.shape(skeleton))
        else:
            embd_lap = 0.
        # --> [batch_size, n_nodes, embd_dim]
        skeleton = tf.concat(tf.unstack(skeleton + embd_lap, axis=0), axis=0)
        # --> [batch_size*n_nodes, embd_dim]
        graph = self.gnn([skeleton, edges_index])
        # graph = tf.nn.leaky_relu(self.gnn_norm(self.gnn([skeleton, edges_index])))
        
        # expand to sequence of graph
        graph = tf.stack(tf.split(graph, self.batch_size, 0), axis=0)
        # [batch_size, n_nodes, embd_dim]
        qvk = self.qvk_conv(graph)
        qvk_seq = tf.stack(tf.split(qvk, self.length, -1), axis=1)
        # [batch_size, length, n_nodes, embd_dim*3]
        qvk_seq = tf.concat(tf.unstack(qvk_seq, axis=-2), -1)
        q, v, k = tf.split(qvk_seq, 3, -1)
        # 3 x [batch_size, length, n_nodes *  embd_dim]
        seq = self.attention([q, v, k])
        seq = tf.stack(tf.split(seq, self.n_nodes, -1), axis=-2)
        # [batch_size, length, n_nodes, embd_dim]
        seq = tf.transpose(seq, [1, 0, 2, 3])
        seq = tf.concat(tf.unstack(seq, axis=1), axis=-2)
        return seq

class ConvShunt(keras.layers.Layer):
    def __init__(self, s_dim, t_dim):
        super(ConvShunt, self).__init__()
        self.to_graph = keras.Sequential()
        for _ in range(3):
            self.to_graph.add(DenseBlock(out_dim=t_dim))
        self.to_g_seq = keras.Sequential()
        for _ in range(3):
            self.to_g_seq.add(DenseConv1D(filters=s_dim, kernel_size=3, strides=1, padding='same'))

    def call(self, inputs, training=None):
        y = self.to_graph(inputs)
        y = tf.expand_dims(y, axis=-1)
        y = self.to_g_seq(y)
        return y

class SeqGraphEmbedding0(keras.layers.Layer):
    '''
    graph positional encoding with Laplacian eigenvectors
    '''
    def __init__(self, length, batch_size, lap_eigvec, embedding_dim):
        super(SeqGraphEmbedding0, self).__init__()
        self.lap_eigvec = lap_eigvec
        self.length = length
        self.batch_size = batch_size
        self.h_embedding_layers = []
        self.h_lap_embedding_layers = []
        for _ in range(length):
            self.h_embedding_layers.append(keras.layers.Dense(embedding_dim))
            self.h_lap_embedding_layers.append(keras.layers.Dense(embedding_dim))

    def call(self, inputs, training=None):
        seq = tf.unstack(inputs, axis=0)
        seq_embd = []
        for i in range(self.length):
            lap_eigvec_embd = self.h_lap_embedding_layers[i](self.lap_eigvec)
            h_embd = self.h_embedding_layers[i](seq[i])
            h_embd = tf.split(h_embd, self.batch_size, axis=0)
            for j in range(self.batch_size):
                h_embd[j] = h_embd[j] + lap_eigvec_embd
            seq_embd.append(tf.concat(h_embd, axis=0))
        seq_embd = tf.stack(seq_embd, axis=0)
        return seq_embd

class SeqGraphEmbedding(keras.layers.Layer):
    '''
    graph positional encoding with Laplacian eigenvectors
    '''
    def __init__(self, length, batch_size, lap_eigvec, embedding_dim):
        super(SeqGraphEmbedding, self).__init__()
        self.lap_eigvec = lap_eigvec
        self.length = length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.h_embedding_layers = []
        self.h_lap_embedding_layers = []
        self.h_embedding_layer = keras.layers.Dense(embedding_dim)
        self.h_lap_embedding_layer = keras.layers.Dense(embedding_dim)

    def call(self, inputs, training=None):
        
        n_nodes = tf.shape(self.lap_eigvec)[0]
        h_embd = self.h_embedding_layer(inputs)
        lap_eigvec_embd = self.h_lap_embedding_layer(self.lap_eigvec)
        lap_eigvec_embd = tf.expand_dims(tf.expand_dims(lap_eigvec_embd, axis=0), axis=0)
        lap_eigvec_embd = tf.broadcast_to(lap_eigvec_embd,
                                          [self.length, self.batch_size, n_nodes, self.embedding_dim])
        lap_eigvec_embd = tf.concat(tf.unstack(lap_eigvec_embd, axis=1), axis=1)
        return lap_eigvec_embd + h_embd

class MULSTM(keras.Model):
    def __init__(self, u, rs=True, use_BiDirect=True, activation='non_linear'):
        super(MULSTM, self).__init__()
        self.use_bidirect = use_BiDirect
        self.activation = activation
        self.layer_forward = keras.layers.LSTM(units=u, 
                                               return_sequences=rs)
        self.ln_f = keras.layers.BatchNormalization()
        if use_BiDirect:
            self.layer_backward = keras.layers.LSTM(units=u, return_sequences=rs, go_backwards=True)
            self.ln_b = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        p = self.ln_f(self.layer_forward(inputs))
        if self.use_bidirect:
            pr = self.ln_b(self.layer_backward(inputs))
            seq = p + pr
        else:
            seq = p
        if self.activation == 'non_linear':
            return tf.nn.leaky_relu(seq)
        elif self.activation == 'linear':
            return seq

class SeqConv1D(keras.Model):
    def __init__(self, u, length, rs=True, activation='non_linear'):
        super(SeqConv1D, self).__init__()
        self.rs = rs
        self.activation = activation
        self.conv1d = keras.layers.Conv1D(filters=u, kernel_size=3, strides=1, padding='same')
        self.norm = keras.layers.LayerNormalization()
        if rs:
            self.gather = keras.layers.Conv1D(filters=u, kernel_size=length, strides=1, padding='valid')
    def call(self, inputs, training=None):
        y = self.conv1d(inputs)
        if self.rs:
            y = self.gather(y)
        y = self.norm(y)
        if self.activation == 'non_linear':
            return tf.nn.leaky_relu(y)
        elif self.activation == 'linear':
            return y

'''
    LSTM-based Models
'''

class GraphLSTMCell(keras.layers.Layer):
    def __init__(self, batch_size, n_nodes, units, n_att_heads=8):
        super(GraphLSTMCell, self).__init__()

        self.units = units
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.gnn_x = GNNBlock(n_layers=1,
                              units=units*4,
                              batch_size=batch_size,
                              n_features=batch_size*n_nodes)
        self.gnn_h = GNNBlock(n_layers=1,
                              units=units*4,
                              batch_size=batch_size,
                              n_features=batch_size*n_nodes)

    def initial_hidden(self):
        return tf.zeros([self.batch_size * self.n_nodes, self.units])

    def call(self, inputs, training=None):
        x = inputs[0]
        e = inputs[1]
        cell = inputs[2]
        h = inputs[3]

        if h is None:
            h = self.initial_hidden()
        if cell is None:
            cell = self.initial_hidden()

        x_gates = self.gnn_x([x, e])
        x_f, x_o, x_i, x_g = tf.split(x_gates, num_or_size_splits=4, axis=-1)
        h_gates = self.gnn_h([h, e])
        h_f, h_o, h_i, h_g = tf.split(h_gates, num_or_size_splits=4, axis=-1)

        i = tf.nn.sigmoid(x_i + h_i)
        f = tf.nn.sigmoid(x_f + h_f)
        updated_cell = f * cell + i * tf.nn.tanh(x_g + h_g)
        o = tf.nn.sigmoid(x_o + h_o)
        updated_h = o * tf.nn.tanh(updated_cell)
        return updated_cell, updated_h

class GraphLSTMLayer(keras.layers.Layer):
    def __init__(self, batch_size, n_nodes, units, n_att_heads=8, rs=True, layernorm=True):
        super(GraphLSTMLayer, self).__init__()
        self.units = units
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.rs = rs
        self.cell = GraphLSTMCell(
                                  batch_size=batch_size,
                                  n_nodes=n_nodes,
                                  units=units,
                                  n_att_heads=n_att_heads
                                  )
        if layernorm and rs:
            self.norm = keras.layers.LayerNormalization(axis=0)
        else:
            self.norm = None

    def call(self, inputs, training=None):
        graph_seq, e_index = inputs[0], inputs[1]
        length = graph_seq.get_shape().as_list()[0]
        graph_list = tf.unstack(graph_seq, length, axis=0)

        out_graph = []
        for i, graph in enumerate(graph_list):
            if i == 0:
                h = None
                c = None
            c, h = self.cell([graph, e_index, c, h])
            out_graph.append(h)
        out_graph = tf.stack(out_graph, axis=0)
        if self.rs:
            if self.norm is None:
                return out_graph
            if self.norm is not None:
                return self.norm(out_graph)
        else:
            return out_graph[-1]

class LSTMEncoder(keras.Model):
    def __init__(self, 
                 lap_eigvec, 
                 edges_index,
                 length, 
                 batch_size, 
                 n_nodes, 
                 dims, 
                 latent_dim, 
                 embedding_dim=16):
        super(LSTMEncoder, self).__init__()

        self.batch_size = batch_size
        self.edges_index = edges_index
        self.gls = []
        for dim in dims[:-1]:
            self.gls.append(GraphLSTMLayer(batch_size=batch_size, n_nodes=n_nodes, units=dim))
        self.last_gl = GraphLSTMLayer(batch_size=batch_size, n_nodes=n_nodes, units=dims[-1], rs=False)

        self.embedding_layer = SeqGraphEmbedding(length=length, 
                                                 batch_size=batch_size,
                                                 lap_eigvec=lap_eigvec, 
                                                 embedding_dim=embedding_dim)
        self.integral = IntegralGraph(
                                      lap_eigvec=lap_eigvec,
                                      batch_size=batch_size,
                                      n_nodes=n_nodes,
                                      latent_dim=latent_dim
                                      )

    def call(self, inputs, training=None):
        y = self.embedding_layer(inputs)
        for layer in self.gls:
            y = layer([y, self.edges_index])
        y = self.last_gl([y, self.edges_index])
        y = self.integral([y, self.edges_index])
        return y
    
class LSTMDecoder(keras.Model):
    def __init__(self, 
                 lap_eigvec, 
                 edges_index,
                 length, 
                 batch_size,
                 n_nodes, 
                 dims, 
                 embedding_dim=16):
        super(LSTMDecoder, self).__init__()
        self.length = length
        self.batch_size = batch_size
        self.edges_index = edges_index
        self.shunt_layer = ShuntLayer(lap_eigvec=lap_eigvec,
                                      batch_size=batch_size,
                                      n_nodes=n_nodes,
                                      length=length,
                                      embedding_dim=embedding_dim)
        self.gls = []
        for dim in dims:
            self.gls.append(GraphLSTMLayer(batch_size=batch_size, n_nodes=n_nodes, units=dim))

    def call(self, inputs, training=None):
        y = self.shunt_layer([inputs, self.edges_index])
        for layer in self.gls:
            y = layer([y, self.edges_index])
        return tf.nn.sigmoid(y)

'''
    TRANSFORMER-based Models
'''

class GraphAttention(keras.layers.Layer):
    def __init__(self, edges_index, length, batch_size, n_nodes, n_head, dim, in_dim):
        super(GraphAttention, self).__init__()
        
        # model hyper parameter variables
        self.length = length
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.edges_index = edges_index
        index_list = []
        for i in range(length):
            ei = edges_index + i * n_nodes
            index_list.append(ei)
        self.seq_edges_index = tf.concat(index_list, axis=1)
        self.n_head = n_head
        self.dim = dim
        self.in_dim = in_dim

        if dim % n_head != 0:
            raise ValueError(
                "dim({}) % n_head({}) is not zero.dim must be multiple of n_head.".format(
                    dim, n_head
                )
            )

        self.d_h = dim // n_head

        self.w_query = GNNBlock(n_layers=1,
                              units=dim,
                              batch_size=batch_size,
                              n_features=batch_size*n_nodes*length)
        self.w_key = GNNBlock(n_layers=1,
                              units=dim,
                              batch_size=batch_size,
                              n_features=batch_size*n_nodes*length)
        self.w_value = GNNBlock(n_layers=1,
                              units=dim,
                              batch_size=batch_size,
                              n_features=batch_size*n_nodes*length)

        self.ff = tf.keras.layers.Dense(dim)

    def call(self, inputs):
        seq_graph = inputs
        seq_graph = self.positional_encoding(seq_graph)
        query = self.seq_graph_gnn(seq_graph, self.seq_edges_index, self.w_query)
        key = self.seq_graph_gnn(seq_graph, self.seq_edges_index, self.w_key)
        value = self.seq_graph_gnn(seq_graph, self.seq_edges_index, self.w_value)

        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        output, _ = self.scaled_dot_product(query, key, value)
        output = self.concat_head(output)

        return tf.transpose(self.ff(output), [1, 0, 2])

    def seq_graph_gnn(self, graph, edges_index, gnn):
        # --- [length, batch_size * n_nodes, in_units]
        seq = tf.concat(tf.unstack(graph, axis=0), axis=0)
        # --> [length * batch_size * n_nodes, in_units]
        y = gnn([seq, edges_index])
        # --> [length * batch_size * n_nodes, dim]
        y = tf.split(y, self.length, 0)
        y = tf.stack(y, axis=1)
        # --> [batch_size * n_nodes, length, dim]
        return y

    def split_head(self, inputs):
        # --- [batch_size * n_nodes, length, dim]
        y = tf.split(inputs, self.n_head, -1)
        # --> n_head x [batch_size * n_nodes, length, d_h] (d_h = dim // n_head)
        y = tf.stack(y, axis=1)
        # --> [batch_size * n_nodes, n_head, length, d_h]
        return y

    def concat_head(self, inputs):
        # --- [batch_size * n_nodes, n_head, length, d_h]
        y = tf.unstack(inputs, axis=1)
        # --> n_head x [batch_size * n_nodes, length, d_h]
        y = tf.concat(y, axis=-1)
        # --- [batch_size * n_nodes, length, dim] (dim = n_head * d_h)
        return y

    def scaled_dot_product(self, query, key, value):
        # --- [batch_size * n_nodes, n_head, length, d_h] for q, k, v
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        # --> [batch_size * n_nodes, n_head, length, length]
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale

        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)
        # --> [batch_size * n_nodes, n_head, length, length]
        attentioned = tf.matmul(attention_weight, value)
        # --> [batch_size * n_nodes, n_head, length, d_h]
        return attentioned, attention_weight
    
    def positional_encoding(self, seq):
        shape = tf.shape(seq)
        pos = np.expand_dims(np.arange(0, self.length), axis=1)
        index = np.expand_dims(np.arange(0, self.in_dim), axis=0)

        pe = pos / np.power(10000, (index - index % 2) / np.float32(self.in_dim))

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=1)
        pe = tf.cast(tf.broadcast_to(pe, shape), dtype=tf.float32)
        seq += pe 
        return seq

class TransformerEncoder(keras.Model):
    def __init__(self, 
                 lap_eigvec, 
                 edges_index, 
                 length, 
                 batch_size, 
                 n_nodes, 
                 dims, 
                 latent_dim,
                 n_head = 1, 
                 embedding_dim = 16):
        super(TransformerEncoder, self).__init__()

        self.edges_index = edges_index
        self.lap_embedding = SeqGraphEmbedding(
                                               length=length,
                                               batch_size=batch_size,
                                               lap_eigvec=lap_eigvec,
                                               embedding_dim=embedding_dim
                                               )
        in_dim_list = [int(embedding_dim)] + dims[:-1]

        self.attlayers = [GraphAttention(
                                         edges_index=edges_index, 
                                         length=length,
                                         batch_size=batch_size,
                                         n_nodes=n_nodes, 
                                         n_head=n_head, 
                                         dim=dims[i],
                                         in_dim=in_dim_list[i]
                                         ) 
                                         for i in range(len(dims))]
        self.integral_to_latent = IntegralLayer(
                                                lap_eigvec=lap_eigvec,
                                                batch_size=batch_size,
                                                n_nodes=n_nodes,
                                                latent_dim=latent_dim,
                                                length=length,
                                                in_dim=dims[-1]
                                                )
        
    def call(self, inputs, training=None):
        y = self.lap_embedding(inputs)
        for attl in self.attlayers:
            y = attl(y)
        y = self.integral_to_latent([y, self.edges_index])
        return y

class TransformerDecoder(keras.Model):
    def __init__(self, 
                 lap_eigvec, 
                 edges_index, 
                 length, 
                 batch_size, 
                 n_nodes, 
                 dims, 
                 n_head = 1, 
                 embedding_dim = 16):
        super(TransformerDecoder, self).__init__()

        self.edges_index = edges_index
        self.shunt_layer = ShuntLayer(lap_eigvec=lap_eigvec,
                                      batch_size=batch_size,
                                      n_nodes=n_nodes,
                                      length=length,
                                      embedding_dim=embedding_dim)
        in_dim_list = [int(embedding_dim)] + dims[:-1]

        self.attlayers = [GraphAttention(
                                         edges_index=edges_index, 
                                         length=length,
                                         batch_size=batch_size,
                                         n_nodes=n_nodes, 
                                         n_head=n_head, 
                                         dim=dims[i],
                                         in_dim=in_dim_list[i]
                                         ) 
                                         for i in range(len(dims))]
        
    def call(self, inputs, training=None):
        y = self.shunt_layer([inputs, self.edges_index])
        for attl in self.attlayers:
            y = attl(y)
        return tf.tanh(y)*1.5

'''
    Merging Models
'''

class MergeLayer(keras.layers.Layer):
    def __init__(self, length, batch_size, n_nodes, edges_index, rs=True, activation='non_linear'):
        super(MergeLayer, self).__init__()
        self.batch_size = batch_size
        self.edges_index = edges_index
        self.rs = rs
        self.lstm_p1 = MULSTM(u=n_nodes, rs=rs, use_BiDirect=True, activation=activation)
        # self.lstm_p1 = SeqConv1D(u=n_nodes, length=length, rs=rs, activation=activation)
        
        # self.gnn_p1 = GNNBlock(n_layers=1, 
        #                       units=length, 
        #                       batch_size=batch_size, 
        #                       n_features=batch_size*n_nodes)
        self.gnn_p1 = GATLayer(units=length,
                               batch_size=batch_size,
                               n_features=batch_size*n_nodes)
        self.lstm_p2 = MULSTM(u=n_nodes, rs=rs, use_BiDirect=True)
        # self.lstm_p2 = SeqConv1D(u=n_nodes, length=length, rs=rs, activation=activation)
        if rs:
            # self.gnn_p2 = GNNBlock(n_layers=1, 
            #                       units=length, 
            #                       batch_size=batch_size, 
            #                       n_features=batch_size*n_nodes,
            #                       activation=activation)
            self.gnn_p2 = GATLayer(units=length, 
                                   batch_size=batch_size, 
                                   n_features=batch_size*n_nodes,
                                   activation=activation)
        else:
            # self.gnn_p2 = GNNBlock(n_layers=1, 
            #                       units=1, 
            #                       batch_size=batch_size, 
            #                       n_features=batch_size*n_nodes,
            #                       activation=activation)
            self.gnn_p2 = GATLayer(units=1, 
                                   batch_size=batch_size, 
                                  n_features=batch_size*n_nodes,
                                  activation=activation)
        
    def call(self, inputs, training=None):
        ### path1: x -> GNN -> LSTM ###
        # -- [batch_size, length, spatial_size]  (inputs shape)
        graph_batch1 = tf.transpose(tf.concat(tf.unstack(inputs, axis=0), axis=-1), [1, 0])
        y_graph = self.gnn_p1([graph_batch1, self.edges_index])
        # -> [batch_size * spatial_size, length]
        seq_batch1 = tf.transpose(tf.stack(tf.split(y_graph, self.batch_size, axis=0), axis=0), [0, 2, 1])
        y_seq1 = self.lstm_p1(seq_batch1)
        # -> [batch_size, length, spatial_size]
        
        ### path2: x -> LSTM -> GNN ###
        seq_batch2 = self.lstm_p2(inputs)
        if self.rs:
            # -> [batch_size, length, spatial_size]
            graph_batch2 = tf.transpose(tf.concat(tf.unstack(seq_batch2, axis=0), axis=-1))
            graph_batch2 = self.gnn_p2([graph_batch2, self.edges_index])
            # -> [batch_size * spatial_size, length]
            y_seq2 = tf.transpose(tf.stack(tf.split(graph_batch2, self.batch_size, axis=0), axis=0), [0, 2, 1])
            # -> [batch_size, length, spatial_size]
        else:
            # -> [batch_size, spatial_size]
            graph_batch2 = tf.concat(tf.unstack(tf.expand_dims(seq_batch2, axis=-1), axis=0), axis=0)
            # -> [batch_size * spatial_size, 1]
            y_seq2 = self.gnn_p2([graph_batch2, self.edges_index])
            y_seq2 = tf.squeeze(tf.stack(tf.split(y_seq2, self.batch_size, axis=0), axis=0))
        
        return y_seq1 + y_seq2

class MergeEncoder(keras.Model):
    def __init__(self, 
                 lap_eigvec, 
                 edges_index,
                 length, 
                 batch_size, 
                 n_nodes, 
                 n_layers, 
                 latent_dim):
        super(MergeEncoder, self).__init__()

        self.batch_size = batch_size
        self.n_layers = n_layers
        self.length = length
        self.n_nodes = n_nodes
        self.edges_index = edges_index
        self.lev = lap_eigvec
        self.gls = [MergeLayer(length=length, 
                               batch_size=batch_size, 
                               n_nodes=n_nodes, 
                               edges_index=edges_index, 
                               ) for _ in range(n_layers)]
        # self.gls.append(MergeLayer(length=length, 
        #                           batch_size=batch_size, 
        #                           n_nodes=n_nodes, 
        #                           edges_index=edges_index, 
        #                           rs=False))
        if lap_eigvec is not None:
            self.embedding_layer = keras.layers.Dense(length)
        # self.integral = IntegralGraph(
        #                               lap_eigvec=lap_eigvec,
        #                               batch_size=batch_size,
        #                               n_nodes=n_nodes,
        #                               latent_dim=latent_dim
        #                               )

        self.rls = [keras.layers.Conv2D(
                                        filters=8, 
                                        kernel_size=[1, 1],
                                        strides=(1, 1),
                                        padding='valid'
                              ) for _ in range(n_layers)]
        self.read_out_norm = keras.layers.BatchNormalization()
        self.out1 = keras.layers.Conv2D(filters=128, 
                                        kernel_size=(3, 3), 
                                        strides=1,
                                        padding='same')
        self.out1_norm = keras.layers.BatchNormalization()
        self.out2 = keras.layers.Conv2D(filters=latent_dim, 
                                        kernel_size=(length, n_nodes), 
                                        strides=1,
                                        padding='valid')

    def call(self, inputs, training=None):
        # [length, batch_size * n_nodes]
        if self.lev is not None:
            y = self.lap_embedding(inputs)
        else:
            y = inputs
        read_outs = []
        for i in range(self.n_layers):
            y = self.gls[i](y)
            read_outs.append(self.rls[i](tf.expand_dims(y, axis=-1)))
        outputs = tf.nn.leaky_relu(self.read_out_norm(tf.concat(read_outs, axis=-1)))
        outputs = tf.nn.leaky_relu(self.out1_norm(self.out1(outputs)))
        outputs = self.out2(outputs)
        # y = tf.concat(tf.unstack(tf.expand_dims(y, axis=-1), axis=0), axis=0)
        # y = self.integral([y, self.edges_index])
        if self.batch_size == 1:
            return tf.expand_dims(tf.squeeze(outputs), axis=0)
        else:
            return tf.squeeze(outputs)

    def lap_embedding(self, inputs):
        # -- [batch_size, length, n_nodes]
        lap_embd = self.embedding_layer(self.lev)
        lap_embd = tf.expand_dims(tf.transpose(lap_embd), axis=0)
        lap_embd = tf.broadcast_to(lap_embd, [self.batch_size, self.length, self.n_nodes])
        return inputs + lap_embd

class MergeDecoder(keras.Model):
    def __init__(self, 
                 lap_eigvec, 
                 edges_index,
                 length, 
                 batch_size,
                 n_nodes, 
                 n_layers):
        super(MergeDecoder, self).__init__()
        self.length = length
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.edges_index = edges_index
        # self.shunt_layer = ShuntLayer(lap_eigvec=lap_eigvec,
        #                               batch_size=batch_size,
        #                               n_nodes=n_nodes,
        #                               length=length)
        self.shunt_layer = ConvShunt(s_dim=n_nodes, t_dim=length)
        self.gls = [MergeLayer(length=length, 
                               batch_size=batch_size, 
                               n_nodes=n_nodes, 
                               edges_index=edges_index) for _ in range(n_layers)]
        self.rls = [keras.layers.Conv2D(
                                        filters=8, 
                                        kernel_size=[1, 1],
                                        strides=(1, 1),
                                        padding='valid'
                              ) for _ in range(n_layers)]
        self.read_out_norm = keras.layers.BatchNormalization()
        self.out1 = ResConv2D(ch=64, strides=1)
        # self.out1 = keras.layers.Conv2D(filters=32, 
        #                                 kernel_size=(3, 3), 
        #                                 strides=1,
        #                                 padding='same')
        # self.out1_norm = keras.layers.BatchNormalization()
        self.out2 = keras.layers.Conv2D(filters=1, 
                                        kernel_size=(1, 1), 
                                        strides=1,
                                        padding='same')

    def call(self, inputs, training=None):
        y = self.shunt_layer(inputs)
        read_outs = []
        read_outs.append(tf.expand_dims(y, axis=-1))
        for i in range(self.n_layers):
            y = self.gls[i](y)
            read_outs.append(self.rls[i](tf.expand_dims(y, axis=-1)))
        outputs = tf.nn.leaky_relu(self.read_out_norm(tf.concat(read_outs, axis=-1)))
        # outputs = tf.nn.leaky_relu(self.out1_norm(self.out1(outputs)))
        outputs = self.out1(outputs)
        outputs = self.out2(outputs)
        if self.batch_size == 1:
            return tf.expand_dims(tf.squeeze(outputs), axis=0)
        else:
            return tf.squeeze(outputs)

    def scale_outputs(self, inputs, nn_spatial, nn_temporal):
        spatial_scaled = nn_spatial(inputs)
        temporal_scaled = nn_temporal(tf.transpose(inputs, [0, 2, 1]))
        temporal_scaled = tf.transpose(temporal_scaled, [0, 2, 1])
        return spatial_scaled + temporal_scaled

class RBF(keras.layers.Layer):
    def __init__(self, rbf_dim, activation='leaky_relu', use_norm=False):
        super(RBF, self).__init__()
        self.activation = activation
        self.use_norm = use_norm
        self.rbf = RandomFourierFeatures(output_dim=rbf_dim, kernel_initializer='gaussian', trainable=True)
        if use_norm:
            self.norm = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        y = self.rbf(inputs)
        if self.use_norm:
            y = self.norm(y)
        if self.activation == 'leaky_relu':
            y = tf.nn.leaky_relu(y)
        if self.activation == 'sigmoid':
            y = tf.nn.sigmoid(y)
        return y

class RBFDense(keras.layers.Layer):
    def __init__(self, dim, activation='non_linear', use_norm=True):
        super(RBFDense, self).__init__()
        self.activation = activation
        self.use_norm = use_norm
        self.rbf = RandomFourierFeatures(output_dim=dim, kernel_initializer='gaussian', trainable=True)
        self.dense = keras.layers.Dense(units=dim)
        if use_norm:
            self.norm = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        kernalized = self.rbf(inputs)
        result = self.dense(kernalized)
        if self.use_norm:
            result = self.norm(result)
        if self.activation == 'non_linear':
            result = tf.nn.leaky_relu(result)
        if self.activation == 'sigmoid':
            result = tf.nn.sigmoid(result)
        if self.activation == 'linear':
            result = result
        return result

class TransNet(keras.Model):
    def __init__(self, dim, hidden_dim=1024):
        super(TransNet, self).__init__()
        self.layer1 = RBF(rbf_dim=hidden_dim, use_norm=True)
        # self.layer2 = RBF(rbf_dim=hidden_dim, use_norm=True)
        # self.layer3 = RBF(rbf_dim=hidden_dim)
        # self.layer4 = RBF(rbf_dim=hidden_dim)
        # self.layer5 = RBF(rbf_dim=hidden_dim)
        # self.layer1 = DenseLayer(hidden_dim)
        # self.layer2 = DenseLayer(hidden_dim)
        # self.layer3 = DenseLayer(hidden_dim)
        # self.layer4 = DenseLayer(hidden_dim)
        # self.layer5 = DenseLayer(hidden_dim)
        self.out = keras.layers.Dense(units=dim)

    def call(self, inputs, training=None):
        y = self.layer1(inputs)
        # y = self.layer2(y)
        # y = self.layer3(y)
        # y = self.layer4(y)
        # y = self.layer5(y)
        y = self.out(y)
        return y

class GraphSeqDiscriminator(keras.Model):
    def __init__(self, ch, lambda_s=0.50, lambda_t=0.50):
        super(GraphSeqDiscriminator, self).__init__()
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.spatial_path = keras.Sequential()
        self.temporal_path = keras.Sequential()
        for i in range(5):
            self.spatial_path.add(Conv1DLayer(filters=ch*2**(i), strides=2, kernel_size=3, padding='same'))
        for i in range(4):
            if i == 3:
                self.spatial_path.add(Conv1DLayer(filters=ch*2**(3-i), strides=1, kernel_size=3, padding='same', activation='linear'))
            else:
                self.spatial_path.add(Conv1DLayer(filters=ch*2**(3-i), strides=1, kernel_size=3, padding='same'))
        for i in range(6):
            self.temporal_path.add(Conv1DLayer(filters=ch*2**(i), strides=2, kernel_size=3, padding='same'))
        for i in range(5):
            if i == 4:
                self.temporal_path.add(Conv1DLayer(filters=ch*2**(4-i), strides=1, kernel_size=3, padding='same', activation='linear'))
            else:
                self.temporal_path.add(Conv1DLayer(filters=ch*2**(4-i), strides=1, kernel_size=3, padding='same'))
        # self.spatial_path.add(ResConv1D(ch=1, strides=1))
        # self.temporal_path.add(ResConv1D(ch=1, strides=1))
        self.spatial_path.add(keras.layers.Flatten())
        self.temporal_path.add(keras.layers.Flatten())
        self.spatial_path.add(keras.layers.Dense(1))
        self.temporal_path.add(keras.layers.Dense(1))
    def call(self, inputs, training=None):
        temporal_inputs = inputs
        spatial_inputs = tf.transpose(inputs, (0, 2, 1))
        y_t = self.temporal_path(temporal_inputs)
        y_s = self.spatial_path(spatial_inputs)
        return self.lambda_s * y_s + self.lambda_t * y_t


class LatentDiscriminator(keras.Model):
    def __init__(self):
        super(LatentDiscriminator, self).__init__()
        self.layer1 = keras.layers.Dense(128)
        self.layer2 = keras.layers.Dense(128)
        self.layer3 = keras.layers.Dense(128)
        self.layer4 = keras.layers.Dense(128)
        self.layer5 = keras.layers.Dense(128)
        self.out = keras.layers.Dense(units=1)
        # self.sn = keras.layers.BatchNormalization()
        
    def call(self, inputs, training=None):
        y = self.layer1(inputs)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.out(y)
        return y


class STConv1D(keras.layers.Layer):
    def __init__(self, ch, strides, activation):
        super(STConv1D, self).__init__()
        self.activation = activation
        self.temporal_conv = tf.keras.layers.Conv1D(filters=ch, kernel_size=3, strides=strides, padding='same')
        self.spatial_conv = tf.keras.layers.Conv1D(filters=ch, kernel_size=3, strides=strides, padding='same')
        self.norm = keras.layers.BatchNormalization()
    def call(self, inputs, trianing=None):
        yt = self.temporal_conv(inputs)
        ys = self.spatial_conv(tf.transpose(inputs, (0, 2, 1)))
        if self.activation == 'linear':
            return self.norm(yt + ys)
        elif self.activation == 'non_linear':
            return tf.nn.leaky_relu(self.norm(yt + ys))


