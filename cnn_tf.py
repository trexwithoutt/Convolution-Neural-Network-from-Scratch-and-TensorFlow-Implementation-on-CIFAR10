import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import numpy as np
import sys
import time

def one_hot(values,n_values=10):
    n_v = np.maximum(n_values, np.max(values) + 1)
    oh=np.eye(n_v)[values]
    return oh

def get_cifar():
    tr=np.float32(np.load('CIFAR_10.npy'))
    tr_lb=np.int32(np.load('CIFAR_labels.npy'))
    tr=tr.reshape((-1,np.prod(np.array(tr.shape)[1:4])))
    train_data=tr[0:45000]/255.
    train_labels=one_hot(tr_lb[0:45000])
    val_data=tr[45000:]/255.
    val_labels=one_hot(tr_lb[45000:])
    
    test_data=np.float32(np.load('CIFAR_10_test.npy'))
    test_data=test_data.reshape((-1,np.prod(np.array(test_data.shape)[1:4])))
    test_data=test_data/255.
    test_labels=one_hot(np.int32(np.load('CIFAR_labels_test.npy')))
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def conv_relu_layer(x,filter_size=[3,3],dim=[1]):

    shape=filter_size+[x.get_shape().as_list()[-1],dim]
    
    W = tf.get_variable('W',shape=shape)
    b = tf.get_variable('b',shape=[dim],initializer=tf.zeros_initializer) 
    
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b)
    return(relu)


def fully_connected_layer(input,dim):
    flat_dim=np.int32(np.array(input.get_shape().as_list())[1:].prod())
    input_flattened = tf.reshape(input, shape=[-1,flat_dim])
    shape=[flat_dim,dim]
    
    W_fc = tf.get_variable('W',shape=shape) 
    b_fc = tf.get_variable('b',shape=[dim],initializer=tf.zeros_initializer)
    
    fc = tf.matmul(input_flattened, W_fc) + b_fc
    return(fc)


def build_cnn(x_image, y_):
    pool_ksize=[1,2,2,1]
    pool_strides=[1,2,2,1]
    with tf.variable_scope("conv1"):
            relu1 = conv_relu_layer(x_image, filter_size=[3, 3],dim=64)
            
    with tf.variable_scope("conv2"):
            relu2 = conv_relu_layer(relu1, filter_size=[3, 3],dim=64)
            pool1 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
    with tf.variable_scope("conv3"):
            relu3 = conv_relu_layer(pool1, filter_size=[3, 3], dim=128)
            
    with tf.variable_scope("conv4"):
            relu4 = conv_relu_layer(relu3, filter_size=[3, 3], dim=128)
            pool2 = tf.nn.max_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
    with tf.variable_scope("conv5"):
            relu5 = conv_relu_layer(pool2, filter_size=[3, 3],dim=256)
            pool3 = tf.nn.max_pool(relu5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
    with tf.variable_scope("fc1"):
            fc1 = fully_connected_layer(pool3, dim=256)
            fc1r=tf.nn.relu(fc1)
   
    with tf.variable_scope("fc2"):
            fc2 = fully_connected_layer(fc1r, dim=10)
            fc2s = tf.nn.softmax(fc2)

    fc2 = tf.identity(fc2, name="OUT")

    with tf.variable_scope('cross_entropy_loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=fc2),name="LOSS")

    with tf.variable_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="ACC")

    return cross_entropy, accuracy, fc2



def get_stats(x, y_, data,labels,fc2):
    t1=time.time()
    loss=0.
    acc=0.
    delta=1000
    rr=np.arange(0, data.shape[0], delta)
    for i in rr:
        fc2_out = fc2.eval(feed_dict={x: data[i:i+delta], y_:labels[i:i+delta]})
        log_sf = logsumexp(fc2_out,axis=1).reshape((fc2_out.shape[0],1)) - fc2_out
        loss += np.mean(np.sum(labels[i:i+delta]*log_sf, axis=1))
        acc += np.mean(np.equal(np.argmax(fc2_out, axis=1),np.argmax(labels[i:i+delta], axis=1)))
    acc = acc/np.float32(len(rr))
    loss = loss/np.float32(len(rr))
    return loss, acc

def run_epoch(x, y_, lr_, step_size, train,val,ii,batch_size,train_step_new):
        t1=time.time()
        # Randomly shuffle the training data
        np.random.shuffle(ii)
        tr=train[0][ii]
        y=train[1][ii]
        lo=0.
        acc=0.
        
        # Run disjoint batches on shuffled data
        for j in np.arange(0,len(y),batch_size):
            batch=(tr[j:j+batch_size],y[j:j+batch_size])
            train_step_new.run(feed_dict={x: batch[0], y_: batch[1], lr_: step_size})
        print('Epoch time',time.time()-t1)

def get_data(data_set):
    if (data_set=="cifar"):
        return(get_cifar())
            
def train(batch_size=500,
          step_size=.0015,
          num_epochs=1,
          num_train=10,
          dim=32,
          nchannels=3,
          optimizer="Adam",
          model_name="model",
          ckpt='./tmp/model.ckpt'):

  x = tf.placeholder(tf.float32, shape=[None, dim*dim*nchannels],name="x")
  x_image = tf.reshape(x, [-1, dim, dim, nchannels])

  y_ = tf.placeholder(tf.float32, shape=[None,10],name="y")

  lr_ = tf.placeholder(tf.float32, shape=[],name="learning_rate")

  loss = []
  vloss= []
  acc =[]
  vacc=[]

  with tf.Session() as sess:
      train,val,test=get_data(data_set="cifar")

      cross_entropy, accuracy, fc2 = build_cnn(x_image, y_)

      # Optimizer
      if (optimizer=="Adam"):
          train_step = tf.train.AdamOptimizer(learning_rate=lr_).minimize(cross_entropy)
      elif (optimizer=="SGD"):
          train_step = tf.train.GradientDescentOptimizer(learning_rate=lr_).minimize(cross_entropy)
      elif (optimizer=="Adagrad"):
          train_step = tf.train.AdagradOptimizer(learning_rate=lr_).minimize(cross_entropy)

      writer = tf.summary.FileWriter("./tmp/output", sess.graph)
      # Initialize variables
      sess.run(tf.global_variables_initializer())

      ii=np.arange(0,num_train,1)

      for i in range(num_epochs):
          run_epoch(x, y_, lr_, step_size,train,val,ii,batch_size,train_step)
          lo,ac = get_stats(x, y_, train[0][0:num_train],train[1][0:num_train], fc2)
          loss.append(lo)
          acc.append(ac)

          vlo,vac = get_stats(x, y_, val[0],val[1], fc2)
          vloss.append(vlo)
          vacc.append(vac)

          if (np.mod(i,10)==0):
              print('Epoch',i,'Train loss, accuracy',lo,ac)
              print('EPoch',i,'Validation loss, accuracy',vlo,vac)
              
      # Save model
      saver = tf.train.Saver()
      save_path = saver.save(sess, ckpt)
      print("Model saved in path: %s" % save_path)
      
      return x, y_, loss, acc, vloss, vacc, cross_entropy, accuracy, saver



def test(accuracy,
         cross_entropy,
         dim=32,
         nchannels=3,
         ckpt='./tmp/model.ckpt'):
  t1=time.time()
  train,val,test=get_data(data_set="cifar")
  with tf.Session() as sess:
    saver.restore(sess, ckpt)
    print('test accuracy %g' % sess.run(accuracy, feed_dict={x: test[0], y_:test[1]}))
    print('test loss %g' % sess.run(cross_entropy, feed_dict={x: test[0], y_:test[1]}))

  print(f'test time: {time.time()-t1}')
  return

# def main():
#     if sys.argv[-1] == '-train':
#         tf.reset_default_graph()
#         x, y_, loss, acc, vloss, vacc, CE, ACCUR, saver = train()

#         plt.plot(loss, label='train loss')
#         plt.plot(vloss, label='validation loss')
#         plt.legend()
#         plt.show()
#         plt.savefig("loss.png")
#         print(np.argmin(np.array(vloss)))

#         plt.plot(acc, label='train accuracy')
#         plt.plot(vacc, label='validation accuracy')
#         plt.legend()
#         plt.show()
#         plt.savefig("acc.png")

#     elif sys.argv[-1] == '-test':
#         test()


if __name__ == '__main__':
    global AC
    global CE
    if sys.argv[1] == '-train':
        tf.reset_default_graph()
        x, y_, loss, acc, vloss, vacc, cross_entropy, accuracy, saver = train()
        AC = accuracy
        CE = cross_entropy
        if sys.argv[-1] == '-test':
            test(AC, CE)
    else:
        print("Model hasn't trained yet")



