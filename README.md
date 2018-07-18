## 简介

适用于Tensorflowjs的**cifar10**的数据集，从[python数据集](https://www.cs.toronto.edu/~kriz/cifar.html)转换而来。

一开始尝试转换为json格式，但是生成后的数据大小要翻了好几倍，目前的方法是像Tensorflowjs官网里的mnist一样，转换成了图片格式，每张图片有一万个样本，每一行代表一个样本，宽度为32 * 32的大小。所以每张图片的尺寸为1024 * 10000

本项目提供了一个Cifar10的类，可以快速的引用并方便的取出里面的数据，并且所有像素已经除以255做好了标准化的工作。



## 安装

#### node平台

```
npm install tfjs-cifar10
```

#### web平台

```
npm install tfjs-cifar10-web
```

注：如果使用的web平台，需要用webpack构建工具，并添加处理json的loader

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

并且要去掉处理png loader里面的 exclude: /node_modules/

```javascript
{
  test: /\.(jpg|jpeg|gif|png)$/,
  // exclude: /node_modules/, <-- 删除这行
  loader: [
  `url-loader?limit=4112&publicPath=./&name=[name].[ext]`
  ]
}
```



## 引用

#### node

```javascript
const { Cifar10 } = require('tfjs-cifar10')
```

注意：如果你用的webpack，你需要把```require```替换成```__non_webpack_require__```

### web

```javascript
import { Cifar10 } from 'tfjs-cifar10-web'
```



## 使用

```javascript
async function load () {
  const data = new Cifar10()
  await data.load()

  const {trainX, trainY} = data.nextTrainBatch(100)
  const {testX, testY} = data.nextTestBatch(1500)
  console.log(trainX, trainY, testX, testY)
  
  const {X, Y} = data.nextTrainBatch() // 如果不传入 默认是取全部数据共五万个样本
}

load()
```

