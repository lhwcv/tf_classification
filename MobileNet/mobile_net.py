import tensorflow as tf
import tensorflow.contrib.slim  as slim
import  slim.nets.mobilenet_v1 as mobilenet_v1

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

##https://stackoverflow.com/questions/42240696/how-could-i-use-batch-normalization-in-tensorflow-slim
def _depthwise_separable_conv(inputs,
                              num_filters,
                              width_multiplier,
                              scope,
                              stride=1):
    num_filters = round(num_filters*width_multiplier)
    depthwise_conv = slim.separable_convolution2d(
        inputs,
        num_outputs=None,# Skip pointwise
        kernel_size=[3,3],
        depth_multiplier=1.0,
        stride=stride,
        scope=scope+'/dw_conv'
    )
    bn = slim.batch_norm(depthwise_conv, scope=scope + '/dw_bn')

    pointwise_conv = slim.convolution2d(bn,num_filters,
                                        kernel_size=[1,1],
                                        scope=scope+'/pw_conv')
    bn = slim.batch_norm(pointwise_conv, scope=scope + '/pw_bn')
    return bn

def mobile_net_inference(images,num_classes,is_training=True,width_multiplier=1.0,scope='MobileNet'):
    """
    MobileNet (https://arxiv.org/abs/1704.04861)
    :param images: input tensor with size [batch_size, h,w,c]
    :param num_classes: number of classes
    :param is_training: flag indicates if is being trained
    :param width_multiplier: width multiplier
    :param scope:
    :return: logits without softmax
    """
    mean_centered_input = tf.to_float(images) -  [_R_MEAN, _G_MEAN,_B_MEAN]
    with tf.variable_scope(scope) as scope:
        end_points = scope.name+'_end_points'
        with slim.arg_scope([slim.convolution2d,slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                decay=0.95,
                                activation_fn=tf.nn.relu
                               ):
               ## https://www.quora.com/Whats-the-difference-between-batch-normalization-and-fused-batch-norm-in-TensorFlow
               ## See Table 1 in the paper

               net = slim.conv2d(mean_centered_input,num_outputs=round(32*width_multiplier),
                                        kernel_size=[3,3], stride=2,scope='conv_1')
               net = slim.batch_norm(net, scope='conv_1/bn')
               net = _depthwise_separable_conv(net, 64, width_multiplier, 'conv_ds2')
               net = _depthwise_separable_conv(net, 128, width_multiplier, 'conv_ds3',stride=2)
               net = _depthwise_separable_conv(net, 128, width_multiplier, 'conv_ds4')
               net = _depthwise_separable_conv(net, 256, width_multiplier, 'conv_ds5',stride=2)
               net = _depthwise_separable_conv(net, 256, width_multiplier, 'conv_ds6')
               net = _depthwise_separable_conv(net, 512, width_multiplier, 'conv_ds7',stride=2)
               net = _depthwise_separable_conv(net, 512, width_multiplier, 'conv_ds8')
               net = _depthwise_separable_conv(net, 512, width_multiplier, 'conv_ds9')
               net = _depthwise_separable_conv(net, 512, width_multiplier, 'conv_ds10')
               net = _depthwise_separable_conv(net, 512, width_multiplier, 'conv_ds11')
               net = _depthwise_separable_conv(net, 512, width_multiplier, 'conv_ds12')
               net = _depthwise_separable_conv(net, 1024, width_multiplier, 'conv_ds13',stride=2)
               net = _depthwise_separable_conv(net, 1024, width_multiplier, 'conv_ds14')
               net = slim.avg_pool2d(net,[7,7],scope='avg_pool15')

        end_points = slim.utils.convert_collection_to_dict(end_points)
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        end_points['squeeze'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
        predictions = slim.softmax(logits, scope='Softmax')

        end_points['Logits'] = logits
        end_points['Predictions'] = predictions

    return logits, end_points




if __name__ =='__main__':
    inputs = tf.random_uniform((8, 224, 224, 3))
    logits,end_points=mobile_net_inference(inputs,2)
    model_variables = slim.get_model_variables(scope='MobileNet')
    logits_slim, end_points_slim = mobilenet_v1.mobilenet_v1(inputs,scope='MobileNetV1')
    model_variables_slim = slim.get_model_variables(scope='MobileNetV1')


    model_variables_maping={}
    for i,j in zip(model_variables,model_variables_slim):
        print(i)
        if 'Logits' not in j.name:
            model_variables_maping[j.name] = i
    print(len(model_variables_maping))









