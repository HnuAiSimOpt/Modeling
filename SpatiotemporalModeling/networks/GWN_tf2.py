import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_geometric as tfg
import time
from networks.customed import ConvSN2D, DenseSN, ConvSN1D

class ResConv2D(keras.Model):
    def __init__(self, ch, strides, padding, activation='non_linear', use_norm=True):
        super(ResConv2D, self).__init__()
        self.strides = strides
        self.activation = activation
        self.use_norm = use_norm
        # self.conv1 = keras.layers.Conv2D(filters=ch, kernel_size=[3, 3],strides=1, padding=padding)
        # self.conv2 = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=strides, padding=padding)
        # self.convid = keras.layers.Conv2D(filters=ch, kernel_size=[1, 1],strides=strides, padding=padding)
        self.conv1 = ConvSN2D(filters=ch, kernel_size=[3, 3],strides=1, padding=padding)
        self.conv2 = ConvSN2D(filters=ch, kernel_size=[1, 1],strides=strides, padding=padding)
        self.convid = ConvSN2D(filters=ch, kernel_size=[1, 1],strides=strides, padding=padding)
        if use_norm:
            self.sn1 = keras.layers.BatchNormalization()
            self.sn2 = keras.layers.BatchNormalization()
            self.sn4 = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        if self.use_norm:
            y = self.sn1(inputs)
        else:
            y = inputs
        y = self.conv1(tf.nn.leaky_relu(y))
        if self.use_norm:
            y = self.sn2(y)
        y = self.conv2(tf.nn.leaky_relu(y))
        if self.use_norm:
            y_id = self.sn4(inputs)
        else:
            y_id = inputs
        identity = self.convid(tf.nn.leaky_relu(y_id))
        if self.activation == 'linear':
            return y + identity
        elif self.activation == 'non_linear':
            return tf.nn.leaky_relu(y + identity)

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

class Conv2DLayer(keras.layers.Layer):
    def __init__(self, ch, strides, kernel_size, padding):
        super(Conv2DLayer, self).__init__()
        self.conv = layers.Conv2D(filters=ch, 
                                kernel_size=kernel_size, 
                                strides=strides,
                                padding=padding)
        self.norm = layers.BatchNormalization()

    def call(self, inputs, training=None):
        return self.norm(self.conv(inputs))

class ConvShunt(keras.layers.Layer):
    def __init__(self, s_dim, t_dim, n_features):
        super(ConvShunt, self).__init__()
        self.N = 1
        self.to_graph = keras.Sequential()
        for _ in range(self.N):
            # self.to_graph.add(DenseBlock(out_dim=t_dim))
            self.to_graph.add(layers.Dense(units=t_dim))
        self.to_g_seq = keras.Sequential()
        for _ in range(self.N):
            # self.to_g_seq.add(DenseConv1D(filters=s_dim, kernel_size=3, strides=1, padding='same'))
            self.to_g_seq.add(layers.Conv1D(filters=s_dim, kernel_size=3, strides=1, padding='same'))
        self.to_g_seq_depth = keras.Sequential()
        for _ in range(self.N):
            # self.to_g_seq_depth.add(ResConv2D(ch=n_features, strides=1, padding='same'))
            self.to_g_seq_depth.add(layers.Conv2D(filters=n_features, strides=1, padding='same', kernel_size=[3, 3]))
        
        # self.to_graph = layers.Dense(units=t_dim)
        # self.to_g_seq = layers.Conv1D(filters=s_dim, kernel_size=3, strides=1, padding='same')
        # self.to_g_seq_depth = layers.Conv2D(filters=n_features, kernel_size=(3, 3), strides=1, padding='same')

    def call(self, inputs, training=None):
        y = self.to_graph(inputs)
        y = tf.expand_dims(y, axis=-1)
        y = self.to_g_seq(y)
        y = tf.expand_dims(y, axis=-1)
        y = self.to_g_seq_depth(y)
        return y

class SNConvShunt(keras.layers.Layer):
    def __init__(self, s_dim, t_dim, n_features):
        super(SNConvShunt, self).__init__()
        self.N = 1
        self.to_graph = keras.Sequential()
        for _ in range(self.N):
            self.to_graph.add(DenseSN(units=t_dim))
        self.to_g_seq = keras.Sequential()
        for _ in range(self.N):
            # self.to_g_seq.add(DenseConv1D(filters=s_dim, kernel_size=3, strides=1, padding='same'))
            self.to_g_seq.add(ConvSN1D(filters=s_dim, kernel_size=3, strides=1, padding='same'))
        self.to_g_seq_depth = keras.Sequential()
        for _ in range(self.N):
            # self.to_g_seq_depth.add(ResConv2D(ch=n_features, strides=1, padding='same'))
            self.to_g_seq_depth.add(ConvSN2D(filters=n_features, strides=1, padding='same', kernel_size=[3, 3]))
        
        # self.to_graph = layers.Dense(units=t_dim)
        # self.to_g_seq = layers.Conv1D(filters=s_dim, kernel_size=3, strides=1, padding='same')
        # self.to_g_seq_depth = layers.Conv2D(filters=n_features, kernel_size=(3, 3), strides=1, padding='same')

    def call(self, inputs, training=None):
        y = self.to_graph(inputs)
        y = tf.expand_dims(y, axis=-1)
        y = self.to_g_seq(y)
        y = tf.expand_dims(y, axis=-1)
        y = self.to_g_seq_depth(y)
        return y

class ShuntLayer(keras.layers.Layer):
    def __init__(self, 
                 n_nodes, 
                 length, 
                 embedding_dim = 1):
        super(ShuntLayer, self).__init__()
        self.length = length
        self.n_nodes = n_nodes
        self.to_grap_skeleton = layers.Dense(n_nodes)
        self.to_graph = layers.Conv1D(filters=embedding_dim, kernel_size=3, strides=1, padding='same')
        self.gnn = tfg.layers.GAT(units=embedding_dim, 
                                  num_heads=1, 
                                  query_activation=tf.nn.leaky_relu, 
                                  key_activation=tf.nn.leaky_relu)
        self.qvk_conv = layers.Conv1D(filters=embedding_dim*3*length, kernel_size=3, strides=1, padding='same')
        self.attention = keras.layers.Attention()
    def call(self, inputs, training=None):
        latent, edges_index = inputs[0], inputs[1]
        bs = latent.get_shape().as_list()[0]
        # expand to graph
        # --- [batch_size, latent_dim]
        skeleton = self.to_grap_skeleton(latent)
        skeleton = tf.expand_dims(skeleton, axis=-1)
        skeleton = self.to_graph(skeleton)
        # --> [batch_size, n_nodes, embd_dim]
        skeleton = tf.concat(tf.unstack(skeleton, axis=0), axis=0)
        # --> [batch_size*n_nodes, embd_dim]
        graph = self.gnn([skeleton, edges_index])
        # graph = tf.nn.leaky_relu(self.gnn_norm(self.gnn([skeleton, edges_index])))
        
        # expand to sequence of graph
        graph = tf.stack(tf.split(graph, bs, 0), axis=0)
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
        return seq
        
class GraphWaveLayer(keras.layers.Layer):
    def __init__(self, ch, edges, seq_length, kernel_size=5):
        super(GraphWaveLayer, self).__init__()
        self.edges = edges
        self.seq_length = seq_length
        n_nodes = int(tf.reduce_max(self.edges))
        self.seq_edges = []
        for i in range(seq_length):
            self.seq_edges.append(self.edges + i*n_nodes)
        self.seq_edges = tf.concat(self.seq_edges, axis=1)

        self.tcn_a = layers.Conv1D(filters=ch, 
                                   kernel_size=kernel_size, 
                                   strides=1, 
                                   padding='causal')
        self.tcn_f = layers.Conv1D(filters=ch, 
                                   kernel_size=kernel_size, 
                                   strides=1, 
                                   padding='causal')
        self.gnn = tfg.layers.GAT(units=ch, 
                                  num_heads=1, 
                                  query_activation=tf.nn.leaky_relu, 
                                  key_activation=tf.nn.leaky_relu)
        self.norm = layers.BatchNormalization()

        self.res_conv = layers.Conv1D(filters=ch, 
                                      kernel_size=1, 
                                      strides=1, 
                                      padding='causal')

    def call(self, inputs, training=None):
        bs = inputs.get_shape().as_list()[0]
        x = tf.transpose(inputs, (0, 2, 1, 3))
        f = self.tcn_f(x)
        f = tf.nn.sigmoid(f)
        # f = tf.nn.leaky_relu(f)
        a = self.tcn_a(x)
        a = tf.nn.tanh(a)
        # a = tf.nn.leaky_relu(a)
        gated = a * f
        # [bs, nodes, seq_len, ch]
        # return gated
        g = self.graph_seq_p_op(gated, self.edges, self.gnn, bs)
        return tf.nn.leaky_relu(self.norm(g + self.res_conv(inputs)))
    
    def graph_seq_p_op(self, x, e, gnn, bs):
        Y = tf.unstack(x, axis=0)
        Y = tf.concat(Y, axis=0)
        Y = tf.unstack(Y, axis=1)
        Y = tf.concat(Y, axis=0)
        Y = gnn([Y, self.seq_edges])
        Y = tf.split(Y, self.seq_length, axis=0)
        Y = tf.stack(Y, axis=0)
        Y = tf.split(Y, bs, axis=1)
        Y = tf.stack(Y, axis=0)
        return Y

class FastGraphWaveLayer(keras.layers.Layer):
    def __init__(self, n_features, n_nodes, edges, seq_length, kernel_size=5, use_norm=True):
        super(FastGraphWaveLayer, self).__init__()
        self.edges = edges
        self.n_features = n_features
        self.seq_length = seq_length
        self.branches = []
        self.use_norm = use_norm
        for _ in range(n_features):
            branch = []
            branch.append(layers.Conv1D(filters=n_nodes, 
                                       kernel_size=kernel_size, 
                                       strides=1,
                                       padding='causal'))
            branch.append(layers.Conv1D(filters=n_nodes, 
                                       kernel_size=kernel_size, 
                                       strides=1, 
                                       padding='causal'))
            # branch.append(layers.LSTM(units=n_nodes, return_sequences=True))
            # branch.append(layers.LSTM(units=n_nodes, return_sequences=True, go_backwards=True))
            branch.append(tfg.layers.GAT(units=seq_length, 
                                      num_heads=1, 
                                      query_activation=tf.nn.leaky_relu, 
                                      key_activation=tf.nn.leaky_relu))
            if use_norm:
                branch.append(layers.BatchNormalization())
            self.branches.append(branch)

        self.res_conv = layers.Conv2D(filters=n_features, 
                                      kernel_size=(1, 1), 
                                      strides=(1, 1), 
                                      padding='valid')

    def call(self, inputs, training=None):
        bs = inputs.get_shape().as_list()[0]
        features = tf.unstack(inputs, axis=-1)
        outputs = []
        for i in range(self.n_features):
            tcn_a = tf.nn.sigmoid(self.branches[i][0](features[i]))
            tcn_b = tf.nn.tanh(self.branches[i][1](features[i]))
            gated = tcn_a * tcn_b

            # tcn_a = self.branches[i][0](features[i])
            # tcn_b = self.branches[i][1](features[i])
            # gated = tcn_a + tcn_b

            graph = tf.concat(tf.unstack(gated, axis=0), axis=-1)
            graph = tf.transpose(graph, [1, 0])
            graph = self.branches[i][2]([graph, self.edges])
            graph = tf.stack(tf.split(graph, bs, axis=0), axis=0)
            graph = tf.transpose(graph, [0, 2, 1])
            graph = tf.expand_dims(graph, axis=-1)
            if self.use_norm:
                graph = self.branches[i][3](graph)
            outputs.append(graph)
        outputs = tf.concat(outputs, axis=-1)
        res = self.res_conv(inputs)
        return outputs + res

class ConvEncoder(keras.Model):
    def __init__(self, 
                ch,
                n_layers, 
                latent_dim,
                use_norm=False):
        super(ConvEncoder, self).__init__()
        self.n_layers = n_layers
        self.use_norm = use_norm
        self.embed = layers.Conv2D(filters=ch, 
                                    kernel_size=(1, 1),
                                    padding='same')
        self.convlayers = []
        for _ in range(n_layers):
            self.convlayers.append(ResConv2D(ch=ch, strides=2, padding='same', use_norm=use_norm))
        self.flat = layers.Flatten()
        self.fc1 = DenseSN(256)
        self.fc2 = DenseSN(latent_dim)
        if use_norm:
            self.bn1 = layers.BatchNormalization()

    def call(self, inputs, training=None):
        y = self.embed(inputs)
        for cl in self.convlayers:
            y = cl(y)
        y = self.flat(y)
        y = self.fc1(y)
        if self.use_norm:
            y = self.bn1(y)
        y = tf.nn.leaky_relu(y)
        y = self.fc2(y)
        return y

class ConvDecoder0(keras.Model):
    def __init__(self, 
                 ch,
                 n_layers, 
                 n_features,
                 seq_length,
                 n_nodes,
                 use_norm=True):
        super(ConvDecoder0, self).__init__()
        self.n_layers = n_layers
        self.shunt = ConvShunt(s_dim=n_nodes, t_dim=seq_length, n_features=1)
        self.gwlayers = []
        for _ in range(n_layers):
            self.gwlayers.append(layers.Conv2D(filters=ch, kernel_size=(3, 3), padding='same'))
        self.skiplayers = []
        for _ in range(n_layers):
            if use_norm:
                self.skiplayers.append(Conv2DLayer(ch=1,
                                                    strides=(1, 1),
                                                    kernel_size=(1, 1),
                                                    padding='valid'))
            else:
                self.skiplayers.append(layers.Conv2D(filters=1,
                                                    strides=(1, 1),
                                                    kernel_size=(1, 1),
                                                    padding='valid'))
        if use_norm:
            self.outlayer1 = Conv2DLayer(ch=1,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        else:
            self.outlayer1 = layers.Conv2D(filters=1,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        self.outlayer2 = layers.Conv2D(filters=n_features,
                                      kernel_size=(1, 1))

    def call(self, inputs, training=None):
        y = self.shunt(inputs)
        # y = tf.expand_dims(y, axis=-1)
        gw_outputs = []
        for i in range(self.n_layers):
            y = self.gwlayers[i](y)
            gw_outputs.append(tf.nn.leaky_relu(self.skiplayers[i](y)))
        gw_outputs = tf.concat(gw_outputs, axis=-1)
        g_Seq = tf.nn.leaky_relu(self.outlayer1(gw_outputs))
        return self.outlayer2(g_Seq)

class Encoder(keras.Model):
    def __init__(self, 
                n_layers, 
                n_features,
                latent_dim,
                edges, 
                seq_length,
                n_nodes,
                use_norm=True,
                n_hidden_feature=4,
                variational=False,
                read_out_dim=1):
        super(Encoder, self).__init__()
        self.variational = variational
        self.n_layers = n_layers
        self.h = latent_dim
        self.embed = layers.Conv2D(filters=read_out_dim, 
                                    kernel_size=(1, 1),
                                    padding='same')
        self.gwlayers = []
        for _ in range(n_layers):
            # self.gwlayers.append(GraphWaveLayer(ch=ch,
            #                                     edges=edges,
            #                                     seq_length=seq_length))
            self.gwlayers.append(FastGraphWaveLayer(n_features=n_hidden_feature, 
                                                    n_nodes=n_nodes, 
                                                    edges=edges, 
                                                    seq_length=seq_length,
                                                    use_norm=use_norm))
        self.skiplayers = []
        for _ in range(n_layers):
            if use_norm:
                self.skiplayers.append(Conv2DLayer(ch=read_out_dim,
                                                   strides=(1, 1),
                                                   kernel_size=[1, 1],
                                                   padding='valid'))
            else:
                self.skiplayers.append(layers.Conv2D(filters=read_out_dim,
                                                    strides=(1, 1),
                                                    kernel_size=(1, 1),
                                                    padding='valid'))
            
        if use_norm:
            self.outlayer1 = Conv2DLayer(ch=read_out_dim,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        else:
            self.outlayer1 = layers.Conv2D(filters=read_out_dim,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        
        if variational:
            self.outlayer2 = layers.Conv2D(filters=32,
                                            kernel_size=(seq_length, n_nodes))
            self.variational_out = layers.Dense(units=latent_dim*2)
        else:
            self.outlayer2 = layers.Conv2D(filters=latent_dim,
                                        kernel_size=(seq_length, n_nodes))

    def call(self, inputs, training=None):
        bs = inputs.get_shape().as_list()[0]
        y = self.embed(inputs)
        gw_outputs = []
        for i in range(self.n_layers):
            y = self.gwlayers[i](y)
            gw_outputs.append(tf.nn.leaky_relu(self.skiplayers[i](y)))
        gw_outputs = tf.concat(gw_outputs, axis=-1)
        latents = tf.nn.leaky_relu(self.outlayer1(gw_outputs))
        latents = self.outlayer2(latents)
        latents = tf.squeeze(latents)
        if bs == 1:
            latents = tf.expand_dims(latents, axis=0)
        if self.variational:
            latents = self.variational_out(latents)
            mean = latents[:, :self.h]
            vari = tf.nn.softplus(latents[:, self.h:])
            return mean, vari
        else:
            return latents

class Decoder(keras.Model):
    def __init__(self, 
                 n_layers, 
                 n_features,
                 edges, 
                 seq_length,
                 n_nodes,
                 use_norm=True,
                 n_hidden_feature=4,
                 read_out_dim=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.edges = edges
        self.shunt = ConvShunt(s_dim=n_nodes, t_dim=seq_length, n_features=n_hidden_feature)
        self.gwlayers = []
        for _ in range(n_layers):
            # self.gwlayers.append(GraphWaveLayer(ch=ch,
            #                                     edges=edges,
            #                                     seq_length=seq_length))
            self.gwlayers.append(FastGraphWaveLayer(n_features=n_hidden_feature, 
                                                    n_nodes=n_nodes, 
                                                    edges=edges, 
                                                    seq_length=seq_length,
                                                    use_norm=use_norm))
        self.skiplayers = []
        for _ in range(n_layers):
            if use_norm:
                self.skiplayers.append(Conv2DLayer(ch=read_out_dim,
                                                    strides=(1, 1),
                                                    kernel_size=(1, 1),
                                                    padding='valid'))
            else:
                self.skiplayers.append(layers.Conv2D(filters=read_out_dim,
                                                    strides=(1, 1),
                                                    kernel_size=(1, 1),
                                                    padding='valid'))
        if use_norm:
            self.outlayer1 = Conv2DLayer(ch=read_out_dim,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        else:
            self.outlayer1 = layers.Conv2D(filters=read_out_dim,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        self.outlayer2 = layers.Conv2D(filters=n_features,
                                      kernel_size=(1, 1))

    def call(self, inputs, training=None):
        y = self.shunt(inputs)
        # y = tf.expand_dims(y, axis=-1)
        gw_outputs = []
        for i in range(self.n_layers):
            y = self.gwlayers[i](y)
            gw_outputs.append(tf.nn.leaky_relu(self.skiplayers[i](y)))
        gw_outputs = tf.concat(gw_outputs, axis=-1)
        g_Seq = tf.nn.leaky_relu(self.outlayer1(gw_outputs))
        return self.outlayer2(g_Seq)

class ConvDecoder(keras.Model):
    def __init__(self, 
                 ch,
                 n_layers, 
                 n_features,
                 seq_length,
                 n_nodes,
                 use_norm=True,
                 n_hidden_feature=1):
        super(ConvDecoder, self).__init__()
        self.n_layers = n_layers
        self.shunt = SNConvShunt(s_dim=n_nodes, t_dim=seq_length, n_features=n_hidden_feature)
        self.gwlayers = []
        for _ in range(n_layers):
            self.gwlayers.append(ResConv2D(ch=ch, strides=1, padding='same'))
        self.outlayer = ResConv2D(ch=n_features, strides=1, padding='same', activation='linear')

    def call(self, inputs, training=None):
        y = self.shunt(inputs)
        # y = tf.expand_dims(y, axis=-1)
        gw_outputs = []
        for i in range(self.n_layers):
            y = self.gwlayers[i](y)
        return self.outlayer(y)

class Discriminator(keras.Model):
    def __init__(self, 
                n_layers, 
                n_features,
                edges, 
                seq_length,
                n_nodes,
                use_norm=False):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        self.embed = layers.Conv2D(filters=n_features, 
                                    kernel_size=(1, 1),
                                    padding='same')
        self.gwlayers = []
        for _ in range(n_layers):
            # self.gwlayers.append(GraphWaveLayer(ch=ch,
            #                                     edges=edges,
            #                                     seq_length=seq_length))
            self.gwlayers.append(FastGraphWaveLayer(n_features=n_features, 
                                                    n_nodes=n_nodes, 
                                                    edges=edges, 
                                                    seq_length=seq_length,
                                                    use_norm=use_norm))
        self.skiplayers = []
        for _ in range(n_layers):
            if use_norm:
                self.skiplayers.append(Conv2DLayer(ch=16,
                                                   strides=(1, 1),
                                                   kernel_size=[1, 1],
                                                   padding='valid'))
            else:
                self.skiplayers.append(layers.Conv2D(filters=16,
                                                    strides=(1, 1),
                                                    kernel_size=(1, 1),
                                                    padding='valid'))
            
        if use_norm:
            self.outlayer1 = Conv2DLayer(ch=64,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        else:
            self.outlayer1 = layers.Conv2D(filters=64,
                                        strides=(1, 1),
                                        kernel_size=(3, 3),
                                        padding='same')
        self.outlayer2 = layers.Conv2D(filters=1,
                                       kernel_size=(seq_length, n_nodes))

    def call(self, inputs, training=None):
        bs = inputs.get_shape().as_list()[0]
        y = self.embed(inputs)
        gw_outputs = []
        for i in range(self.n_layers):
            y = self.gwlayers[i](y)
            gw_outputs.append(tf.nn.leaky_relu(self.skiplayers[i](y)))
        gw_outputs = tf.concat(gw_outputs, axis=-1)
        latents = tf.nn.leaky_relu(self.outlayer1(gw_outputs))
        latents = self.outlayer2(latents)
        latents = tf.squeeze(latents)
        if bs != 1:
            return latents
        else:
            return tf.expand_dims(latents, axis=0)


if __name__ == '__main__':
    input_shape = [10, 5]
    e = tf.random.uniform([2, 500], maxval=240, dtype=tf.int32)
    x = tf.random.normal(input_shape)
    # fast_gw = FastGraphWaveLayer(n_features=3, n_nodes=240, edges=e, seq_length=200)
    # y = fast_gw(x)
    # print(tf.shape(y))
    # gat = tfg.layers.GAT(units=4, 
    #                     num_heads=1, 
    #                     query_activation=tf.nn.leaky_relu, 
    #                     key_activation=tf.nn.leaky_relu)
    # y = gat([x, e])
    # print(tf.shape(y))



    # gw_layer = GraphWaveLayer(ch=64, edges=e, seq_length=200)
    # st = time.time()
    # y = gw_layer(x)
    # et = time.time()
    
    decoder = ConvDecoder(ch=64,
                          n_layers=4,
                          n_features=1,
                          seq_length=200,
                          n_nodes=240)
    encoder = ConvEncoder(ch=64,
                          n_layers=4,
                          latent_dim=1)
    # encoder = Encoder(
    #                 n_layers=1,
    #                 n_features=1,
    #                 latent_dim=5,
    #                 edges=e,
    #                 seq_length=200,
    #                 n_nodes=240,
    #                 n_hidden_feature=1,
    #                 variational=True
    #                 )
    # decoder = Decoder(
    #                 n_layers=6,
    #                 n_features=3,
    #                 edges=e,
    #                 seq_length=200,
    #                 n_nodes=240)
    l = decoder(x)
    y = encoder(l)
    print(tf.shape(l), tf.shape(y))


