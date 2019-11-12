from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import random
import collections
import math
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.contrib.layers import xavier_initializer_conv2d


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)

current_version = '1'
dataset_name = 'UKDALE'
appliance_name = 'WashingMachine'
appliance_name_folder = '%s_%s' %(dataset_name, appliance_name)
base_root = 'outnilm\\%s\\%s' % (appliance_name_folder, current_version)
folder_list = {
    'base_root': base_root,
    'image': '%s\\image' % base_root,
    'model': '%s\\model' % base_root,
    'test_validation': '%s\\test_validation' % base_root,
    'input': 'data\\UK_DALE\\%s' % appliance_name,
}

# if there's no checkpoint, it will be 1
isFirst = 1
parser = argparse.ArgumentParser()
if isFirst == 1:
    parser.add_argument("--checkpoint",
                        default=None,
                        help="directory with checkpoint to resume training from or use for testing")
else:
    parser.add_argument("--checkpoint",
                        default=folder_list['model'],
                        help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--output_dir",
                    default=folder_list['base_root'],
                    help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", default=120, type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=1000, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=5000, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int,
                    default=500,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int,
                    default=50000,
                    help="save model every save_freq steps, 0 to disable")
parser.add_argument('--test_procedure', type=int, default=50000)
parser.add_argument('--validation_procedure', type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--lr_d", type=float, default=0.001, help="initial learning rate for sgd")
parser.add_argument("--lr_g", type=float, default=0.0005, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

a = parser.parse_args()
EPS = 1e-12
gggg = 0
scale = 1

Examples = collections.namedtuple("Examples", "inputs, targets, count,"
                                              "inputs_test, targets_test, count_test,"
                                              "steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, discrim_loss,"
                                        "gen_loss_GAN, gen_loss_L1, train")


def discrim_conv(batch_input, out_channels, stride):
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=(4, 1), strides=stride, padding="same",
                                kernel_initializer=xavier_initializer_conv2d())


def gen_conv(batch_input, out_channels):
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=(4, 1), strides=(2, 1), padding="same",
                            kernel_initializer=xavier_initializer_conv2d())


def gen_deconv(batch_input, out_channels):
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=(4, 1), strides=(2, 1), padding="same",
                                      kernel_initializer=xavier_initializer_conv2d())


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(x, scope='batch_instance_norm'):
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0),
                              constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,
        a.ngf * 4,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.0),
        (a.ngf * 4, 0.0),
        (a.ngf * 2, 0.0),
        (a.ngf, 0.0),
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            rectified = tf.nn.relu(input)
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets, step):
    learning_rate_d = tf.train.exponential_decay(a.lr_d, step, 50000, 0.95, staircase=True)
    learning_rate_g = tf.train.exponential_decay(a.lr_g, step, 50000, 0.95, staircase=True)

    # def create_discriminator(discrim_inputs, discrim_targets):
    #     n_layers = 3
    #     layers = []
    #
    #     input = tf.concat([discrim_inputs, discrim_targets], axis=1)
    #     with tf.variable_scope("layer_1"):
    #         convolved = discrim_conv(input, a.ndf, stride=(2, 1))
    #         rectified = lrelu(convolved, 0.2)
    #         layers.append(rectified)
    #     for i in range(n_layers):
    #         with tf.variable_scope("layer_%d" % (len(layers) + 1)):
    #             out_channels = a.ndf * min(2**(i+1), 8)
    #             stride = 1 if i == n_layers - 1 else (2, 1)
    #             convolved = discrim_conv(layers[-1], out_channels, stride=stride)
    #             normalized = batchnorm(convolved)
    #             rectified = lrelu(normalized, 0.2)
    #     with tf.variable_scope("layer_%d" % (len(layers) + 1)):
    #         convolved = discrim_conv(rectified, out_channels=1, stride=1)
    #         output = tf.sigmoid(convolved)
    #         layers.append(output)
    #
    #     return layers[-1]
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=1)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=(2, 1))
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else (2, 1)  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)[:, 256:768, :, :]

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_d)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(learning_rate=learning_rate_g, beta1=a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        discrim_loss=ema.average(discrim_loss),
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def display_image(inputs, targets, outputs):
    global gggg
    fig = plt.figure(figsize=(16, 16), dpi=100)
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.05, hspace=0.05)
    agg = np.reshape(inputs, [1024])
    app = np.reshape(targets, [512])
    out = np.reshape(outputs, [512])
    agg_new = []
    app_new = []
    out_new = []
    x = np.linspace(256, 768, 512)

    for i in agg:
        agg_new.append(int(i*scale))
    for i in range(512):
        app_new.append(int(app[i]*scale))
        out_new.append(int(out[i]*scale))

    plt.subplot(3, 1, 1)
    plt.xlim(0, 1024)
    plt.plot(agg_new)

    plt.subplot(3, 1, 2)
    plt.xlim(0, 1024)
    plt.plot(x, app_new)

    plt.subplot(3, 1, 3)
    plt.xlim(0, 1024)
    plt.plot(x, out_new)

    plt.savefig(('%s/{}.png' % folder_list['image'])
                .format(str(gggg).zfill(3)), bbox_inches='tight')
    gggg += 1
    plt.close(fig)


def metric(target_data, output_data, step, epoch):
    tar_new = []
    out_new = []
    mae = []

    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            if output_data[i][j] < 0:
                o = 0
            else:
                o = output_data[i][j]
            out_new.append(o)
            tar_new.append(target_data[i][j])

    for i in range(len(out_new)):
        r = abs(tar_new[i] - out_new[i])
        mae.append(r)

    accuracy = (1.0-(np.sum(mae, dtype=np.int64)/(2*np.sum(tar_new, dtype=np.int64))))*100
    print('np.sum(mae):')
    print(np.sum(mae))
    print('np.sum(tar_new):')
    print(np.sum(tar_new))
    sae = abs(np.sum(tar_new, dtype=np.int64)-np.sum(out_new, dtype=np.int64))/np.sum(tar_new, dtype=np.int64)
    mse = mean_squared_error(tar_new, out_new)
    mae = mean_absolute_error(tar_new, out_new)
    r2 = r2_score(tar_new, out_new)

    with open('%s\\test.txt' % folder_list['test_validation'], 'a+') as f:
        print('STEP: %d' % (step+1), file=f)
        print('Epoch: %d' % epoch, file=f)
        print('Accuracy: %f' % accuracy, file=f)
        print('SAE: %f' % sae, file=f)
        print('MSE: %f' % mse, file=f)
        print('MAE: %f' % mae, file=f)
        print('R^2: %f' % r2, file=f)
        print('', file=f)


def load_examples():
    base_path = folder_list['input']

    ap_data = np.load(os.path.join(base_path, 'appliance_train_1.npy'))
    ag_data = np.load(os.path.join(base_path, 'main_train_1.npy'))
    ap_data_test = np.load(os.path.join(base_path, 'appliance_test_1.npy'))
    ag_data_test = np.load(os.path.join(base_path, 'main_test_1.npy'))
    print('Data load complete!')

    appliance_data = tf.cast(ap_data, tf.float32)
    aggregate_data = tf.cast(ag_data, tf.float32)
    appliance_data_test = tf.cast(ap_data_test, tf.float32)
    aggregate_data_test = tf.cast(ag_data_test, tf.float32)
    print('cast complete!')

    queue = tf.train.slice_input_producer([aggregate_data, appliance_data], shuffle=True)
    inputs_batch, targets_batch = tf.train.batch(queue, batch_size=a.batch_size)
    queue_test = tf.train.slice_input_producer([aggregate_data_test, appliance_data_test], shuffle=False)
    inputs_batch_test, targets_batch_test = tf.train.batch(queue_test, batch_size=a.batch_size)

    steps_per_epoch = int(math.ceil(len(ap_data) / a.batch_size))

    return Examples(
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(ap_data),
        inputs_test=inputs_batch_test,
        targets_test=targets_batch_test,
        count_test=len(ap_data_test),
        steps_per_epoch=steps_per_epoch,
    )


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    for folder_name, folder_path in folder_list.items():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    print('Folder creation complete!')

    with open(os.path.join(folder_list['base_root'], 'config.txt'), 'w') as f:
        for k, v in a._get_kwargs():
            print(k, "=", v, file=f)
    print('Config save complete!')

    examples = load_examples()
    global_step = tf.placeholder(tf.float32, None)
    inputs = tf.placeholder(tf.float32, [a.batch_size, 1024, 1, 1])
    targets = tf.placeholder(tf.float32, [a.batch_size, 512, 1, 1])
    i_ = tf.reshape(examples.inputs, [a.batch_size, 1024, 1, 1])
    t_ = tf.reshape(examples.targets, [a.batch_size, 512, 1, 1])
    i_t = tf.reshape(examples.inputs_test, [a.batch_size, 1024, 1, 1])
    t_t = tf.reshape(examples.targets_test, [a.batch_size, 512, 1, 1])
    model = create_model(inputs, targets, global_step)
    print('model complete!')

    with tf.name_scope("display_images"):
        display_fetches = {
            'output': model.outputs,
        }

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=50)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("parameter_count =", sess.run(parameter_count))
        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        start = time.time()
        output_test = []
        target_test = []

        for step in range(max_steps):

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }
            if should(a.progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            inputs_, targets_ = sess.run([i_, t_])
            results = sess.run(fetches,
                               options=options,
                               run_metadata=run_metadata,
                               feed_dict={global_step: step,
                                          inputs: inputs_,
                                          targets: targets_})

            if should(a.display_freq):
                data = results['display']
                display_image(inputs_, targets_, data['output'])

            if should(a.summary_freq):
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(a.progress_freq):
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                print("progress  epoch %d  step %d  image/sec %0.1f" %
                      (train_epoch, train_step+1, rate))

                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(folder_list['model'], "model"), global_step=sv.global_step)

            if should(a.test_procedure):
                print('Test procedure!')
                target_test.clear()
                output_test.clear()
                max_steps_test = examples.count_test
                for step_test in range(max_steps_test):
                    inputs_, targets_= sess.run([i_t, t_t])
                    data = sess.run({
                        'output': model.outputs,
                    }, feed_dict={
                        global_step: results['global_step'],
                        inputs: inputs_,
                        targets: targets_
                    })
                    _target = np.reshape(targets_, [a.batch_size, 512])
                    _output = np.reshape(data['output'], [a.batch_size, 512])
                    for i in range(a.batch_size):
                        tar_temp = []
                        out_temp = []
                        for j in range(512):
                            tar_temp.append(int(_target[i][j] * scale))
                            out_temp.append(int(_output[i][j] * scale))
                        target_test.append(tar_temp)
                        output_test.append(out_temp)
                    if step_test % 1000 == 0:
                        print('Process Test: %f  Rate: %f' % (step_test / max_steps_test,
                                                              (time.time() - start) / max_steps_test))
                np.save('%s\\test_target_%d.npy' % (folder_list['test_validation'], results['global_step']+1), target_test)
                np.save('%s\\test_output_%d.npy' % (folder_list['test_validation'], results['global_step']+1), output_test)
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                metric(target_test, output_test, results['global_step'],train_epoch)
                print('Test procedure complete!')

            if sv.should_stop():
                break


main()
