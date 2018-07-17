import * as tf from '@tensorflow/tfjs'
import { Image, createCanvas } from 'canvas'
import fs from 'fs'
import path from 'path'

export class DataSet {
  readonly IMG_WIDTH: number
  readonly IMG_HEIGHT: number
  readonly TRAIN_IMAGES: string[]
  readonly TRAIN_LABLES: string
  readonly TEST_IMAGES: string[]
  readonly TEST_LABLES: string
  readonly NUM_CLASSES: number
  readonly DATA_PRE_NUM: number = 10000

  get IMAGE_SIZE () {
    return this.IMG_WIDTH * this.IMG_HEIGHT * 3
  }

  trainDatas: Float32Array[]
  testDatas: Float32Array[]
  trainLables: number[]
  testLables: number[]

  trainM: number = 0
  testM: number = 0
  trainIndices: Uint32Array
  testIndices: Uint32Array
  shuffledTrainIndex: number = 0;
  shuffledTestIndex: number = 0

  currentTrainIndex: number = 0

  loadImg (src: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      const canvas = createCanvas()
      const ctx = canvas.getContext('2d') as CanvasRenderingContext2D

      fs.readFile(src, (err, squid) => {
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
    return Promise.all(srcs.map(this.loadImg))
    // .then(async imgsBytesView => imgsBytesView
    //   .reduce((preView, currentView) => this.float32Concat(preView, currentView)))
  }

  async load () {
    this.trainDatas = await this.loadImages(this.TRAIN_IMAGES)
    this.testDatas = await this.loadImages(this.TEST_IMAGES)

    this.trainLables = await JSON.parse(fs.readFileSync(this.TRAIN_LABLES, 'utf8'))
    this.testLables = await JSON.parse(fs.readFileSync(this.TEST_LABLES, 'utf8'))

    this.trainM = this.trainLables.length
    this.testM = this.testLables.length

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(this.trainM)
    this.testIndices = tf.util.createShuffledIndices(this.testM)
  }

  nextBatch (batchSize: number, [data, lables]: [Float32Array[], number[]], index: Function) {
    const batchImagesArray = new Float32Array(batchSize * this.IMAGE_SIZE)
    const batchLabelsArray = new Uint8Array(batchSize * this.NUM_CLASSES)

    const batchLables = []

    for (let i = 0; i < batchSize; i++) {
      const idx = index()
      const currentIdx = idx % this.DATA_PRE_NUM
      const dataIdx = Math.floor(idx / this.DATA_PRE_NUM)

      const image =
          data[dataIdx].slice(currentIdx * this.IMAGE_SIZE, currentIdx * this.IMAGE_SIZE + this.IMAGE_SIZE)
      batchImagesArray.set(image, i * this.IMAGE_SIZE)
      batchLables.push(lables[idx])
    }
    const xs = tf.tensor2d(batchImagesArray, [batchSize, this.IMAGE_SIZE])
    const ys = tf.oneHot(batchLables, this.NUM_CLASSES)

    return { xs, ys }
  }

  nextTrainBatch (batchSize: number = this.trainM) {
    this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length

    return this.nextBatch(
      batchSize, [this.trainDatas, this.trainLables], () => {
        this.shuffledTrainIndex =
            (this.shuffledTrainIndex + 1) % this.trainIndices.length
        return this.trainIndices[this.shuffledTrainIndex]
      })
  }

  nextTestBatch (batchSize: number = this.testM) {
    this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length

    return this.nextBatch(
      batchSize, [this.testDatas, this.testLables], () => {
        this.shuffledTestIndex =
            (this.shuffledTestIndex + 1) % this.testIndices.length
        return this.testIndices[this.shuffledTestIndex]
      })
  }
}
