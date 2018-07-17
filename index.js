const { Cifar10 } = require('./dist/node/index.bundle.js')

async function  lode () {
  const data = new Cifar10()
  await data.load()
}

lode()