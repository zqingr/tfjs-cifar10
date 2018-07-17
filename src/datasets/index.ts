import { DataSet } from './base'

export class Cifar10 extends DataSet {
  TRAIN_IMAGES = [
    require('./data_batch_1.png'),
    require('./data_batch_2.png'),
    require('./data_batch_3.png'),
    require('./data_batch_4.png'),
    require('./data_batch_5.png')
  ]
  TRAIN_LABLES = require('./train_lables.json')
  TEST_IMAGES = [
    require('./test_batch.png')
  ]
  TEST_LABLES = require('./test_lables.json')
  IMG_WIDTH = 32
  IMG_HEIGHT = 32
  NUM_CLASSES = 10
}
