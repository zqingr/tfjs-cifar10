English | [中文](./README_cn.md)


## Introduction

Dataset for **cifar10** for Tensorflowjs. Converted from [python](https://www.cs.toronto.edu/~kriz/cifar.html).

I initially tried to convert to the json format, but the generated data is much larger. The current method is to convert to the image format like the mnist in the official website of Tensorflowjs.  Each picture has 10,000 samples, and each line represents one. The sample has a width of 32 * 32. So the size of each image is 1024 * 10000.

This project provides a Cifar10 class that can quickly reference and easily retrieve the data inside, and all pixels have been divided by 255 to do the standardization work.

## Installation

#### Node platform

```
npm install tfjs-cifar10
```

#### Web platform

```
npm install tfjs-cifar10-web
```

Note: If you are using a web platform, you need to use the webpack build tool and add a loader that handles json.

```javascript
...
{
  type: 'javascript/auto',
  test: /\.(json)$/,
  exclude: /node_modules/,
  loader: [
    `file-loader?publicPath=./&name=[name].[ext]`
  ]
}
...
```

And to remove the exclude inside the png loader: /node_modules/

```javascript
{
  test: /\.(jpg|jpeg|gif|png)$/,
  // exclude: /node_modules/, <-- Delete this line
  loader: [
  `url-loader?limit=4112&publicPath=./&name=[name].[ext]`
  ]
}
```



## Import

#### node

```javascript
const { Cifar10 } = require('tfjs-cifar10')
```

Note: If you use webpack, you need to replace ```require``` with ```__non_webpack_require__```

### web

```javascript
import { Cifar10 } from 'tfjs-cifar10-web'
```



## How to use

```javascript
async function load () {
  const data = new Cifar10()
  await data.load()

  const {trainX, trainY} = data.nextTrainBatch(100)
  const {testX, testY} = data.nextTestBatch(1500)
  console.log(trainX, trainY, testX, testY)
  
  const {X, Y} = data.nextTrainBatch() // If you do not pass in the parameters, the default is to take a total of 50,000 samples of all data.
}

load()
```

## Demo

 - [cifar10_cnn](https://github.com/zqingr/tfjs-examples/tree/master/src/examples/cifar10_cnn)
 - [cifar10_resnet](https://github.com/zqingr/tfjs-examples/tree/master/src/examples/cifar10_resnet)
