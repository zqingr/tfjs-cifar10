import { DataSet } from './datasets/base'
import { Image, createCanvas } from 'canvas'
import * as tf from '.0.12.2@@tensorflow/tfjs/dist'
import fs from 'fs'
import path from 'path'

export class Cifar10 extends DataSet {
  TRAIN_IMAGES = [
    require('./datasets/data_batch_1.png'),
    require('./datasets/data_batch_2.png'),
    require('./datasets/data_batch_3.png'),
    require('./datasets/data_batch_4.png'),
    require('./datasets/data_batch_5.png')
  ]
  TRAIN_LABLES = require('./datasets/train_lables.json')
  TEST_IMAGES = [
    require('./datasets/test_batch.png')
  ]
  TEST_LABLES = require('./datasets/test_lables.json')

  getPath (src: string) {
    return path.join(path.dirname(__filename), src)
  }

  loadImg (src: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      const canvas = createCanvas()
      const ctx = canvas.getContext('2d') as CanvasRenderingContext2D

      fs.readFile(this.getPath(src), (err, squid) => {
        if (err) throw err
        const img = new Image()
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
        img.src = squid
      })
    })
  }

  loadImages (srcs: string[]): Promise<Float32Array[]> {
    return Promise.all(srcs.map((src) => this.loadImg(src)))
    // .then(async imgsBytesView => imgsBytesView
    //   .reduce((preView, currentView) => this.float32Concat(preView, currentView)))
  }

  async load () {
    this.trainDatas = await this.loadImages(this.TRAIN_IMAGES)
    this.testDatas = await this.loadImages(this.TEST_IMAGES)

    this.trainLables = await JSON.parse(fs.readFileSync(this.getPath(this.TRAIN_LABLES), 'utf8'))
    this.testLables = await JSON.parse(fs.readFileSync(this.getPath(this.TEST_LABLES), 'utf8'))

    this.trainM = this.trainLables.length
    this.testM = this.testLables.length

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(this.trainM)
    this.testIndices = tf.util.createShuffledIndices(this.testM)
  }
}
