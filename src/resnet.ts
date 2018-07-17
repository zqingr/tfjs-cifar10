/**
  Trains a ResNet on the CIFAR10 dataset.
  ResNet v1
  [a] Deep Residual Learning for Image Recognition
  https://arxiv.org/pdf/1512.03385.pdf
  ResNet v2
  [b] Identity Mappings in Deep Residual Networks
  https://arxiv.org/pdf/1603.05027.pdf
*/

import * as tf from '@tensorflow/tfjs'

/**
 * 2D Convolution-Batch Normalization-Activation stack builder
 *
 * @param param0
 * @param inputs (SymbolicTensor): input from input image or previous layer
 * @param num_filters (number): Conv2D number of filters
 * @param kernel_size (number): Conv2D square kernel dimensions
 * @param strides (number): Conv2D square stride dimensions
 * @param activation (string): activation name
 * @param batch_normalization (boolean): whether to include batch normalization
 * @param conv_first (boolean): conv-bn-activation (True) or
 * @returns x (SymbolicTensor): SymbolicTensor as input to the next layer
*/
function resnetLayer (
  { inputs, filters = 16, kernelSize = 3, strides = 1, activation = 'relu', batchNormalization = true, convFirst = true }:
  { inputs: tf.SymbolicTensor, filters?: number, kernelSize?: number, strides?: number, activation?: string, batchNormalization?: boolean, convFirst?: boolean }
): tf.SymbolicTensor {
  const conv = tf.layers.conv2d({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.heNormal({}), kernelRegularizer: tf.regularizers.l2({l2: 1e-4}) })
  let x = inputs
  if (convFirst) {
    x = conv.apply(x) as tf.SymbolicTensor
    if (batchNormalization) x = tf.layers.batchNormalization({axis: 3}).apply(x) as tf.SymbolicTensor
    if (activation) x = tf.layers.activation({ activation }).apply(x) as tf.SymbolicTensor
  } else {
    if (batchNormalization) x = tf.layers.batchNormalization({axis: 3}).apply(x) as tf.SymbolicTensor
    if (activation) x = tf.layers.activation({ activation }).apply(x) as tf.SymbolicTensor
    x = conv.apply(x) as tf.SymbolicTensor
  }
  return x
}

/**
 * ResNet Version 1 Model builder [a]
 *
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
 *
 * @param input_shape (number[]): shape of input image tensor
 * @param depth (number): number of core convolutional layers
 * @param numClasses (number): number of classes (CIFAR10 has 10)
 * @returns model (Model): tfjs model instance
 */
export function resnetV1 ({ inputShape, depth, numClasses = 10 }: {inputShape: number[], depth: number, numClasses?: number}) {
  if ((depth - 2) % 6 !== 0) throw new Error('depth should be 6n+2 (eg 20, 32, 44 in [a])')

  // Start model definition.
  let numFilters = 16
  const numResBlocks = (depth - 2) / 6

  const inputs = tf.input({ shape: inputShape })
  let x = resnetLayer({ inputs })

  // Instantiate the stack of residual units
  for (let stack = 0; stack < 3; stack++) {
    for (let resBlock = 0; resBlock < numResBlocks; resBlock++) {
      const isFirstLayerNotFirstStack = stack > 0 && resBlock === 0
      const strides = isFirstLayerNotFirstStack ? 2 : 1
      let y = resnetLayer({ inputs: x, filters: numFilters, strides })
      y = resnetLayer({ inputs: y, filters: numFilters, activation: '' })

      if (isFirstLayerNotFirstStack) {
        // linear projection residual shortcut connection to match
        // changed dims
        x = resnetLayer({ inputs: x, filters: numFilters, kernelSize: 1, strides, activation: '', batchNormalization: false })
      }
      x = tf.layers.add().apply([x, y]) as tf.SymbolicTensor
      x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor
    }
    numFilters *= 2
  }

  // Add classifier on top.
  // v1 does not use BN after last shortcut connection-ReLU
  x = tf.layers.averagePooling2d({ poolSize: 8 }).apply(x) as tf.SymbolicTensor
  const y = tf.layers.flatten().apply(x) as tf.SymbolicTensor
  const outputs = tf.layers.dense({ units: numClasses, activation: 'softmax', kernelInitializer: tf.initializers.heNormal({}) }).apply(y) as tf.SymbolicTensor

  return tf.model({ inputs: inputs, outputs, name: 'ResNetV1_' + depth })
}

/**
 * ResNet Version 2 Model builder [b]
 *
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
 *
 * @param input_shape (number[]): shape of input image tensor
 * @param depth (number): number of core convolutional layers
 * @param numClasses (number): number of classes (CIFAR10 has 10)
 * @returns model (Model): tfjs model instance
 */
export function resnetV2 ({ inputShape, depth, numClasses = 10 }: {inputShape: number[], depth: number, numClasses?: number}) {
  if ((depth - 2) % 9 !== 0) throw new Error('depth should be 9n+2 (eg 56 or 110 in [b])')

  // Start model definition.
  let numFiltersIn = 16
  let numFiltersOut = 0
  const numResBlocks = (depth - 2) / 9

  const inputs = tf.input({ shape: inputShape })
  let x = resnetLayer({ inputs })

  // Instantiate the stack of residual units
  for (let stack = 0; stack < 3; stack++) {
    for (let resBlock = 0; resBlock < numResBlocks; resBlock++) {
      let batchNormalization = true
      let strides = 1
      let activation = 'relu'

      if (stack === 0) { // first stack
        numFiltersOut = numFiltersIn * 4
        if (resBlock === 0) { // first layer and first stage
          activation = ''
          batchNormalization = false
        }
      } else {
        numFiltersOut = numFiltersIn * 2
        if (resBlock === 0) { // first layer but not first stage
          strides = 2
        }
      }

      let y = resnetLayer({ inputs: x, filters: numFiltersIn, kernelSize: 1, strides, activation, batchNormalization, convFirst: false }) as tf.SymbolicTensor
      y = resnetLayer({ inputs: y, filters: numFiltersIn, convFirst: false }) as tf.SymbolicTensor
      y = resnetLayer({ inputs: y, filters: numFiltersOut, kernelSize: 1, convFirst: false }) as tf.SymbolicTensor

      if (resBlock === 0) {
        x = resnetLayer({ inputs: x, filters: numFiltersOut, kernelSize: 1, strides, activation: '', batchNormalization: false })
      }
      x = tf.layers.add().apply([x, y]) as tf.SymbolicTensor
    }
    numFiltersIn = numFiltersOut
  }
  // Add classifier on top.
  // v2 has BN-ReLU before Pooling
  x = tf.layers.batchNormalization({}).apply(x) as tf.SymbolicTensor
  x = tf.layers.activation({ activation: 'relu' }).apply(x) as tf.SymbolicTensor
  x = tf.layers.averagePooling2d({ poolSize: 8 }).apply(x) as tf.SymbolicTensor
  let outputs = tf.layers.flatten({}).apply(x) as tf.SymbolicTensor
  outputs = tf.layers.dense({ units: numClasses, activation: 'softmax', kernelInitializer: tf.initializers.heNormal({}) }).apply(outputs) as tf.SymbolicTensor

  return tf.model({ inputs: inputs, outputs, name: 'ResNetV2_' + depth })
}
