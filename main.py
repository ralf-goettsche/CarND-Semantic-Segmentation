import os.path
import numpy as np
import tensorflow as tf
import scipy.misc
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import shutil
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Training Parameter
# KTTI :  289 elements
#
# Settings for 2 classes
EPOCHS = 56
BATCH_SIZE = 5
KEEP_PROB = 0.75
LEARN_RATE = 0.001
#
# Settings for 3 classes
#EPOCHS = 200
#BATCH_SIZE = 10
#KEEP_PROB = 0.75
#LEARN_RATE = 0.001

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Loading VGG-16 graph
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # Defining graph to get tensors for 'skipping layer operation' 
    detect_graph = tf.Graph()
    detect_graph = tf.get_default_graph()
    
    # Reading out tensors of VGG-16
    detect_image_input = detect_graph.get_tensor_by_name(vgg_input_tensor_name)
    detect_keep_prob = detect_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    detect_layer3_out = detect_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    detect_layer4_out = detect_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    detect_layer7_out = detect_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return (detect_image_input, detect_keep_prob, detect_layer3_out, detect_layer4_out, detect_layer7_out)
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network fcn-8.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # Defining inference graph using conv2d_1x1 to adopt feature depth to number of classes depth
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=(1, 1), strides=(1,1), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="conv_1x1_layer7")
    seg_sem_trans_1 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="seg_sem_trans_1")
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=(1, 1), strides=(1,1), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="conv_1x1_layer4")
    seg_sem_merge_1 = tf.add(seg_sem_trans_1, conv_1x1_layer4, name="seg_sem_merge_1")
    seg_sem_trans_2 = tf.layers.conv2d_transpose(seg_sem_merge_1, num_classes, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="seg_sem_trans_2")
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=(1, 1), strides=(1,1), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="conv_1x1_layer3")
    seg_sem_merge_2 = tf.add(seg_sem_trans_2, conv_1x1_layer3, name="seg_sem_merge_2")
    seg_sem_trans_3_out = tf.layers.conv2d_transpose(seg_sem_merge_2, num_classes, kernel_size=(16, 16), strides=(8, 8), padding="same",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="seg_sem_trans_3_out")
    return (seg_sem_trans_3_out)
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Transforming 4-D tensors to 2-D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    
    # Softmax-cross-entropy loss calculation  
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label))
    
    # training with Adam and using softmax-cross-entropy loss calculation
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return (logits, train_op, cross_entropy_loss)
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Initializing variables
    sess.run(tf.global_variables_initializer())
    
    print("Training ...")
    print()

    # Training fully-connected network (VGG-16 + inference graph) with global parameter KEEP_PROB and LEARN_RATE
    min_loss = 999.9
    losses = []
    for i in range(epochs):
        batch_losses = []
        for image, label in get_batches_fn(batch_size):
            sess.run(train_op, feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARN_RATE})
            loss = sess.run(cross_entropy_loss, feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB})
            if loss < min_loss:
                min_loss = loss
            batch_losses.append(loss)

        losses.append(batch_losses)
        print("EPOCH {} ...".format(i+1))
        print("Loss  = ",loss)
        print("(temp. Min-Loss  = {:.3f})".format(min_loss))
        print()

    print("END Training:")
    print("Min-Loss  = {:.3f}".format(min_loss))
    print()
    
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)
    plt.title("Loss Development")
    plt.ylabel("Error")

    # Create the boxplot
    plt.style.use('fivethirtyeight')
    del losses[0]
    bp = ax.boxplot(losses)
    
    plt.savefig("./loss_plot.png")
    
    #plt.show()

    pass
tests.test_train_nn(train_nn)


def gen_mov_pic(image, sess, logits, input_image, keep_prob, image_shape):
    """
    Generating semantic segementation pictures (used by movie generation).
    Taken from helper.py:get_test_output() and adopted (no loading of images and no label return)
    :param image: input image (frame from movie) to be segmented
    :param sess: TF Session
    :param logits: TF unnormalized log probabilities of the output layer
    :param input_image: TF Placeholder for input images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param image_shape: shape of the image (frame)
    """

    # Reshaping image to target dim
    image = scipy.misc.imresize(image, image_shape)
    # Running fcn on image (frame)
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_image: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)
    
def gen_3class_mov_pic(image, sess, logits, input_image, keep_prob, image_shape):
    """
    Generating semantic segementation pictures (used by movie generation).
    Taken from helper.py:get_test_output() and adopted (no loading of images and no label return)
    :param image: input image (frame from movie) to be segmented
    :param sess: TF Session
    :param logits: TF unnormalized log probabilities of the output layer
    :param input_image: TF Placeholder for input images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param image_shape: shape of the image (frame)
    """

    # Reshaping image to target dim
    image = scipy.misc.imresize(image, image_shape)
    # Running fcn on image (frame)
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_image: [image]})
    # Calculation of filter
    im_class1 = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    im_class2 = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
    segmenclass1 = (im_class1 > 0.5).reshape(image_shape[0], image_shape[1], 1)
    segmenclass2 = (im_class2 > 0.5).reshape(image_shape[0], image_shape[1], 1)
    # Generation of green colored mask for street in own direction
    mask_street = np.dot(segmenclass1, np.array([[0, 255, 0, 127]]))
    mask_street = scipy.misc.toimage(mask_street, mode="RGBA")
    # Generation of blue colored mask for street in counter direction
    mask_cntstr = np.dot(segmenclass2, np.array([[0, 0, 255, 127]]))
    mask_cntstr = scipy.misc.toimage(mask_cntstr, mode="RGBA")
    # Adding mask to image
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask_street, box=None, mask=mask_street)
    street_im.paste(mask_cntstr, box=None, mask=mask_cntstr)

    return np.array(street_im)
    
def gen_mov_pic_wrapper(movie, sess, logits, input_image, keep_prob, image_shape, num_classes):
# Workaround to manipulate movie frames by a function with any parameter
# (VideoFileClip.fl_image() only allows call of a function with the only parameter image):
# Using wrapper function with all the needed parameters which returns the VideoFileClip operation fl_image calling 
# a core function. The wrapper function is around this core function having image as the only parameter but 
# returning the call of the image (frame) manipulation function with all the needed parameters.
# [Taken and adopted from https://github.com/Zulko/moviepy/issues/159 ]

    def gen_mov_pic_caller(image):
        return gen_mov_pic(image, sess, logits, input_image, keep_prob, image_shape)

    def gen_3class_mov_pic_caller(image):
        return gen_3class_mov_pic(image, sess, logits, input_image, keep_prob, image_shape)

    if num_classes == 2:
        return movie.fl_image(gen_mov_pic_caller)
    elif num_classes == 3:
        return movie.fl_image(gen_3class_mov_pic_caller)
    else:
        print("INFO: Movie generation for num_classes > 3 isn't implemented!")
        exit()
    pass



def run():
    
    num_classes = 2
    #num_classes = 3
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
   
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/


    # Settings to control what should be done
    train = False    # Training of model and saving it; if set to False, a model will be loaded
    img_gen = False # Perform image test (given by project)
    mov_gen = True  # Perform semantic segmentation on movie
    
    # Definition of path to model, its directory, and its name
    model_dir_name = "vgg_sem_seg__classes_{}_epochs_{}_batch_{}_keep_{}_learn_{}_augm".format(num_classes, EPOCHS, BATCH_SIZE, KEEP_PROB, LEARN_RATE)
    model_path = "./model/vgg_sem_seg/"+model_dir_name
    model_name = "vgg_sem_seg"
    
    # Definition of path to movie and its new name after semantic segementation
    moviepath = "./project_video.mp4"
    movieoutpath = "./project_video_segmented.mp4"

    with tf.Session() as sess:
    
        if train:
            print()
            print("Building model ...")
            print()
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            if (num_classes == 2):
                get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
            elif (num_classes == 3):
                get_batches_fn = helper.gen_3class_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
            else:
                print ("ERROR: There is no function defined for num_classes={}! Abbortion!".format(num_classes))
                exit(0)
            
            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # Building VGG-16 detection graph
            (image_input, keep_prob, layer3_out, layer4_out, layer7_out) = load_vgg(sess, vgg_path)
            
            # Building FCN-8 for semantic segmentation
            output = layers(layer3_out, layer4_out, layer7_out, num_classes)
            
            # Defining training and optimization strategy
            (logits, train_op, cross_entropy_loss) = optimize(output, correct_label, learning_rate, num_classes)
            
            # Train FCN
            train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
            
            # Saving model und model_path
            print("Saving model under {} ...".format(model_path))
            print()
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            os.makedirs(model_path)

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(model_path, model_name))
            print("Model saved.")
            print()
            
        else:
            # If no model training, a model will be loaded and appropriate tensors defined
            print()
            print("Loading model {} ...".format(os.path.join(model_path, model_name)))
            print()
            get_saver = tf.train.import_meta_graph(os.path.join(model_path, model_name)+".meta")
            get_saver.restore(sess, os.path.join(model_path, model_name))
            
            # Definition of appropriate tensors
            graph = tf.get_default_graph()
            image_input = graph.get_tensor_by_name('image_input:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            output = graph.get_tensor_by_name('seg_sem_trans_3_out/conv2d_transpose:0')
            logits = tf.reshape(output, (-1, num_classes))

            print("Model loaded.")
            print()

                 
        if img_gen:
            # Testing model on test images and storing the result under ./run
            print("Testing model on images ...")
            print()
            if (num_classes == 2):
                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
            elif (num_classes == 3):
                helper.save_3class_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
            else:
                print ("ERROR: There is no function defined for num_classes={}! Abbortion!".format(num_classes))
                exit(0)
            print("Test done.")
            print()

        
        # OPTIONAL: Apply the trained model to a video
        if mov_gen:
            # Apply FCN on movie
            print("Generating movie ...")
            print()
            movie = VideoFileClip(moviepath)
            segmented_movie = movie.fx(gen_mov_pic_wrapper, sess, logits, image_input, keep_prob, image_shape, num_classes)
            segmented_movie.write_videofile(movieoutpath, audio=False)
            print("Movie generation done.")
            print()


if __name__ == '__main__':
    run()
