const tf = __non_webpack_require__('@tensorflow/tfjs')

export class DataSet {
  readonly IMG_WIDTH: number = 32
  readonly IMG_HEIGHT: number = 32
  readonly TRAIN_IMAGES: string[]
  readonly TRAIN_LABLES: string
  readonly TEST_IMAGES: string[]
  readonly TEST_LABLES: string
  readonly NUM_CLASSES: number = 10
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

  getPath (src: string) {
  }

  loadImg (src: string) {
  }

  loadImages (srcs: string[]) {
  }

  async load () {
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
