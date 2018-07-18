import { DataSet } from './datasets/base'
import * as tf from '@tensorflow/tfjs'

export class Cifar10 extends DataSet {
  TRAIN_IMAGES = [
    __non_webpack_require__('./data_batch_1.png'),
    __non_webpack_require__('./data_batch_2.png'),
    __non_webpack_require__('./data_batch_3.png'),
    __non_webpack_require__('./data_batch_4.png'),
    __non_webpack_require__('./data_batch_5.png')
  ]
  TRAIN_LABLES = __non_webpack_require__('./train_lables.json')
  TEST_IMAGES = [
    __non_webpack_require__('./test_batch.png')
  ]
  TEST_LABLES = __non_webpack_require__('./test_lables.json')

  loadImg (src: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
      img.src = src
      img.onload = () => {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight

        const datasetBytesBuffer =
            new ArrayBuffer(canvas.width * canvas.height * 3 * 4)
        const datasetBytesView = new Float32Array(datasetBytesBuffer)

        ctx.drawImage(
          img, 0, 0, canvas.width, canvas.height, 0, 0, canvas.width,
          canvas.height)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        for (let j = 0, i = 0; j < imageData.data.length; j++) {
          if ((j + 1) % 4 === 0) continue
          datasetBytesView[i++] = imageData.data[j] / 255
        }

        resolve(datasetBytesView)
      }
      img.onerror = reject
    })
  }

  loadImages (srcs: string[]): Promise<Float32Array[]> {
    return Promise.all(srcs.map(this.loadImg))
    // .then(async imgsBytesView => imgsBytesView
    //   .reduce((preView, currentView) => this.float32Concat(preView, currentView)))
  }

  async load () {
    this.trainDatas = await this.loadImages(this.TRAIN_IMAGES)
    this.testDatas = await this.loadImages(this.TEST_IMAGES)

    this.trainLables = await fetch(this.TRAIN_LABLES).then(res => res.json())
    this.testLables = await fetch(this.TEST_LABLES).then(res => res.json())

    this.trainM = this.trainLables.length
    this.testM = this.testLables.length

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(this.trainM)
    this.testIndices = tf.util.createShuffledIndices(this.testM)
  }
}
