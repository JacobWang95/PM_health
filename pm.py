import tensorflow as tf
import numpy as np 
import cPickle as pickle 
#import tf_util
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
#parser.add_argument('--model', default='lanenet', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--max_epoch', type=int, default=1000000, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir

num_feat = 271
num_mid = 10

batch_size = BATCH_SIZE

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp pm.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate    

def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def optim(loss, category, lr=0.0001, beta1=0.9, beta2=0.99):
    optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)

    var_list = [t for t in tf.trainable_variables() if t.name.startswith(category)]
    gradient = optim.compute_gradients(loss, var_list = var_list)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optim.apply_gradients(gradient, global_step=global_step), global_step

def placeholder_input(batch_size, num_feat):
    place_input = tf.placeholder(tf.float32, shape=(batch_size, num_feat))
    place_output = tf.placeholder(tf.float32, shape=(batch_size, 1))
    return place_input, place_output

def cal_loss(gt, loss):
    num_all = gt.get_shape()[0].value
    num_pos = tf.reduce_sum(gt)
    num_neg = num_all - num_pos
    #loss_weight = gt * ((num_neg) / (num_pos) - 1) + 1.0
    loss_weight = gt * (10 - 1) + 1.0
    loss_f = tf.reduce_sum(loss*loss_weight)
    #tf.Print(num_pos, data)
    return loss_f

def get_model(input_ph, gt_ph):
    batch_size = input_ph.get_shape()[0].value
    with tf.variable_scope('main_net'):        
        weight_l1 = tf.get_variable('weight_l1', [num_feat, num_mid])#, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        layer_1 = tf.tanh(tf.matmul(input_ph, weight_l1))

        weight_o = tf.get_variable('weight_o', [num_mid, 1])#, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        layer_o = tf.sigmoid(tf.matmul(layer_1, weight_o))

        weight_sum = tf.reduce_sum(tf.abs(weight_l1))

        #loss_rc_raw = (layer_o-gt_ph)*(layer_o-gt_ph)
        loss_rc_raw = -(gt_ph * tf.log(layer_o) + (1-gt_ph) * tf.log(1-layer_o))
        loss_rc = cal_loss(gt_ph, loss_rc_raw)

    with tf.variable_scope('pm_net'):
        shut_mat = tf.constant(1 - np.eye(num_mid), dtype = np.float32)
        weight_dict = {}
        pm_dict = {}
        #pm_out_dict = {}
        pm_loss_dict = {}
        loss_pm_sum = tf.Variable(0.0, trainable=False)
        for i in range(num_mid):
            weight_dict['weight_pm_'+str(i)] = tf.get_variable('weight_pm_'+str(i), [num_mid,1])#, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
            pm_dict['pm_dict_'+str(i)] = tf.matmul(layer_1*shut_mat[i], weight_dict['weight_pm_'+str(i)])
            pm_loss_dict['pm_loss' + str(i)] = tf.reduce_mean((pm_dict['pm_dict_'+str(i)] - layer_1[i])*(pm_dict['pm_dict_'+str(i)] - layer_1[i])) 
            loss_pm_sum += pm_loss_dict['pm_loss' + str(i)]
    loss_main = loss_rc + 0.15* weight_sum - loss_pm_sum#/num_mid
    tf.summary.scalar('loss_main', loss_main)
    tf.summary.scalar('loss_pm', loss_pm_sum)
    tf.summary.scalar('loss_rc', loss_rc)
    tf.summary.scalar('loss_l1', weight_sum)
    tf.summary.histogram('weight_dis', weight_l1)
    return loss_main, loss_pm_sum, layer_o, layer_1

def train_one_epoch(input_data, gt_data, sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    #is_training = True
    
    # Shuffle train files
    #train_file_idxs = np.arange(0, len(TRAIN_FILES))
    #np.random.shuffle(train_file_idxs)
    
    log_string('----' + str('Hahahahaha') + '-----')
    current_data, current_label, _ = shuffle_data(input_data, gt_data)            
    #current_label = np.squeeze(current_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size / BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        in_data = current_data[start_idx:end_idx, :, :]
        in_gt = current_label[start_idx:end_idx, :, :]

        summary, step, _, loss_val,l_matrix,t_matrix = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['matrix'],ops['trans_ma']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val


    log_string('loss: %f' % (loss_sum/num_batches))


with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        input_ph, gt_ph = placeholder_input(batch_size, num_feat)

        loss_main, loss_pm_sum, layer_o, layer_1 = get_model(input_ph, gt_ph)


        train_main, main_global_step = optim(loss_main, category = 'main_net')
        train_pm, pm_global_step = optim(loss_pm_sum, category = 'pm_net')

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        sess = tf.Session(config = config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        #input_data = pickle.load()
        #gt_data = pickle.load()
        #input_data = np.random.random((400,280))
        #gt_data = np.random.random((400,1))
        data_all = np.load('./data/data.npy')
        input_data = data_all[0:1400000, 0:-1]
        gt_data = np.reshape(data_all[0:1400000, -1],(-1, 1))

        input_test = data_all[-148208:, 0:-1]
        gt_test = np.reshape(data_all[-148208:, -1],(-1, 1))
        file_size = input_data.shape[0]
        num_batches = file_size / BATCH_SIZE
        sess.run(init)
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            current_data, current_label, _ = shuffle_data(input_data, gt_data)            
            
            file_size = current_data.shape[0]
            num_batches = file_size / BATCH_SIZE
            loss_sum = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE

                in_data = current_data[start_idx:end_idx, ...]
                in_gt = current_label[start_idx:end_idx, ...]
                
                _, main_step, loss_val = sess.run([train_main, main_global_step, loss_main], feed_dict={input_ph: in_data, gt_ph: in_gt})
                _, pm_step, summary = sess.run([train_pm, pm_global_step, merged], feed_dict={input_ph: in_data, gt_ph: in_gt})
                #summary  = sess.run([merged])


                train_writer.add_summary(summary, main_step)
                loss_sum += loss_val
            ind_test = np.random.randint(0, 148208 - batch_size)
            re_test = sess.run([layer_o], feed_dict={input_ph: input_test[ind_test:ind_test+batch_size, ...], gt_ph: gt_test[ind_test:ind_test+batch_size, ...]})

            for ind_loop in range(batch_size):
                print [re_test[0][ind_loop], gt_test[ind_test + ind_loop]]
            #print re_test
            #print gt_test[ind_test:ind_test+100, ...]


            log_string('loss: %f' % (loss_sum/num_batches))

            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


