
import tensorflow as tf
import numpy as np 
import cPickle as pickle 
#import tf_util
import argparse
import os
import sys
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
#parser.add_argument('--model', default='lanenet', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--max_epoch', type=int, default=1000000, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='log21/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=30000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
MODEL_PATH = FLAGS.model_path
task_rc = 1

num_feat = 140
if task_rc == 1: num_feat_c = num_feat
if task_rc == 0: num_feat_c = 1
num_mid = 10

batch_size = BATCH_SIZE

os.system('cp pm2.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def optim(loss, category, lr=0.001, beta1=0.9, beta2=0.99):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
                        lr,  # Base learning rate.
                        global_step, #* BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)
    tf.summary.scalar('learning_rate', learning_rate)
    optim = tf.train.AdamOptimizer(learning_rate=learing_rate, beta1=beta1, beta2=beta2)

    var_list = [t for t in tf.trainable_variables() if t.name.startswith(category)]
    gradient = optim.compute_gradients(loss, var_list = var_list)

    return optim.apply_gradients(gradient, global_step=global_step), global_step

def placeholder_input(batch_size, num_feat):
    place_input = tf.placeholder(tf.float32, shape=(batch_size, num_feat))
    place_output = tf.placeholder(tf.float32, shape=(batch_size, num_feat_c))
    return place_input, place_output

def cal_loss(gt, loss):
#    num_all = batch_size * num_feat_c
#    num_pos = tf.reduce_sum(gt)
#    num_neg = num_all - num_pos
#    
#    if task_rc == 0: 
#        loss_weight = gt * ((num_neg) / (num_pos) - 1) + 1.0
#    elif task_rc == 1:

    loss_w = []
    #loss_weight = tf.constant(loss_w)
    for idd in range(num_feat_c):
        num_pos = tf.reduce_sum(gt[:, idd])
        num_neg = batch_size - num_pos
        loss_w.append(tf.transpose(gt[:, idd] * ((num_neg) / (num_pos+0.00001) - 1) + 1.0))
    loss_weight = tf.transpose(tf.reshape(tf.stack(loss_w), (num_feat_c, batch_size)))
    #loss_weight = tf.constant(loss_w)
    #loss_weight = gt * (5 - 1) + 1.0
    loss_mat = loss * loss_weight
    loss_f = tf.reduce_mean(loss_mat)
    #tf.Print(num_pos, data)
    return loss_f, loss * loss_weight

def get_model(input_ph, gt_ph):
    batch_size = input_ph.get_shape()[0].value
    with tf.variable_scope('main_net'):        
        weight_l1 = tf.get_variable('weight_l1', [num_feat, num_mid], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        bias_l1 = tf.get_variable('bias_l1', [num_mid],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        layer_1 = tf.tanh(tf.matmul(input_ph, weight_l1) + bias_l1)

        weight_o = tf.get_variable('weight_o', [num_mid, num_feat_c], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        bias_o = tf.get_variable('bias_o', [num_feat_c],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        layer_o = tf.sigmoid(tf.matmul(layer_1, weight_o) + bias_o)
        #layer_o = tf.matmul(layer_1, weight_o) + bias_o

        weight_sum = tf.reduce_sum(tf.abs(weight_l1))

        loss_rc_raw = (layer_o-gt_ph)*(layer_o-gt_ph)
        loss_rc, loss_mat = cal_loss(gt_ph, loss_rc_raw)
        #loss_mat = -(gt_ph * tf.log(layer_o) + (1 - gt_ph) * tf.log(1 - layer_o))
        #loss_rc = tf.reduce_mean(loss_mat)

    with tf.variable_scope('pm_net'):
        shut_mat = tf.constant(1 - np.eye(num_mid), dtype = np.float32)
        weight_dict = {}
        bias_dict = {}
        pm_dict = {}
        #pm_out_dict = {}
        pm_loss_dict = {}
        loss_pm_sum = tf.Variable(0.0, trainable=False)
        for i in range(num_mid):
            weight_dict['weight_pm_'+str(i)] = tf.get_variable('weight_pm_'+str(i), [num_mid,1])#, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
            bias_dict['bias_pm_'+str(i)] = tf.get_variable('bias_pm_'+str(i), [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            pm_dict['pm_dict_'+str(i)] = tf.tanh(tf.matmul(layer_1*shut_mat[i], weight_dict['weight_pm_'+str(i)]) + bias_dict['bias_pm_'+str(i)])
            #pm_dict['pm_dict_'+str(i)] = tf.matmul(layer_1*shut_mat[i], weight_dict['weight_pm_'+str(i)]) + bias_dict['bias_pm_'+str(i)]
            pm_loss_dict['pm_loss' + str(i)] = tf.reduce_mean((pm_dict['pm_dict_'+str(i)] - layer_1[i])*(pm_dict['pm_dict_'+str(i)] - layer_1[i]))
            loss_pm_sum += pm_loss_dict['pm_loss' + str(i)]

    loss_main = 200 * loss_rc + 1 * weight_sum - 0.2 * loss_pm_sum#/num_mid

    tf.summary.scalar('loss_main', loss_main)
    tf.summary.scalar('loss_pm', loss_pm_sum)
    tf.summary.scalar('loss_rc', loss_rc)
    tf.summary.scalar('loss_l1', weight_sum)
    tf.summary.histogram('weight_dis', weight_l1)
    return loss_main, loss_rc, loss_pm_sum, layer_o, layer_1, loss_mat, weight_o, weight_l1


with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        input_ph, gt_ph = placeholder_input(batch_size, num_feat)

        loss_main, loss_rc, loss_pm_sum, layer_o, layer_1, loss_mat, weight_o, weight_l1 = get_model(input_ph, gt_ph)

        #main_global_step = tf.Variable(0, trainable=False)
        #learning_rate = get_learning_rate(main_global_step)
        #tf.summary.scalar('learning_rate', learning_rate)

        #train_main, main_global_step = optim(loss_main, category = 'main_net')#, lr = learning_rate)
        #train_pm, pm_global_step = optim(loss_pm_sum, category = 'pm_net')#, lr = learning_rate)

        #batch = main_global_step
        #tf.summary.scalar('batch', batch)

        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    sess = tf.Session(config = config)
    
    #init = tf.global_variables_initializer()

    saver.restore(sess, MODEL_PATH)

    data_all = np.load('./data/data_w.npy')
    input_data = data_all[0:1400000, 0:-1]
    if task_rc == 0: 
        gt_data = np.reshape(data_all[0:1400000, -1],(-1, 1))
    elif task_rc == 1: 
        gt_data = input_data

    input_test = data_all[-148208:, 0:-1]
    if task_rc == 0: 
        gt_test = np.reshape(data_all[-148208:, -1],(-1, 1))
    elif task_rc == 1: 
        gt_test = input_test

    file_size = input_data.shape[0]
    num_batches = file_size / BATCH_SIZE
    #sess.run(init)
    w1, w2 = sess.run([weight_l1, weight_o], feed_dict={input_ph: input_data[:5000, ...], gt_ph: gt_data[:5000, ...]})

    model_p = MODEL_PATH.split('/')[0]
    np.save(model_p + '/w1', w1)
    np.save(model_p + '/w2', w2)
