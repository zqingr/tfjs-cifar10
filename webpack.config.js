const path = require('path');

module.exports = function (env) {
  const PLATFORM = process.env.PLATFORM
  const type = PLATFORM === 'node' ? 'node' : 'web'
  return {
    mode: 'production',
    context: path.join(process.cwd(), 'src'),
    target: type,
    entry: {
      index: './index_' + type + '.ts'
    },
    output: {
      path: path.resolve(__dirname, 'dist', type),
      filename: '[name].bundle.js',
      libraryTarget: "commonjs"
    },
    resolve: {
      // Add `.ts` and `.tsx` as a resolvable extension.
      extensions: [".ts", ".tsx", ".js"]
    },
    module: {
      rules: [
      {
        test: /\.(ts)$/,
        enforce: "pre",
        exclude: /node_modules/,
        loader: "eslint-loader",
        options: {
          fix: true
        }
      }, 
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        loader: 'ts-loader',
        options: {
          appendTsSuffixTo: [/\.vue$/, /\.ts$/],
          transpileOnly: true
        }
      },
      {
        test: /\.html$/,
        loader: 'raw-loader',
        exclude: ['./src/index.html']
      },
      {
        test: /\.(jpg|jpeg|gif|png)$/,
        exclude: /node_modules/,
        loader: [
          `url-loader?limit=4112&publicPath=./&name=[name].[ext]`
        ]
      },
      {
        type: 'javascript/auto',
        test: /\.(json)$/,
        exclude: /node_modules/,
        loader: [
          `file-loader?publicPath=./&name=[name].[ext]`
        ]
      },
      {
        test: /\.css$/,
        exclude: /node_modules/,
        use: [
          'style-loader',
          'css-loader'
        ]
      }]
    },
    node: {
      __filename: false,
      __dirname: false
    },
    externals: {
      '@tensorflow/tfjs': '@tensorflow/tfjs',
      '@tensorflow/tfjs-node': '@tensorflow/tfjs-node',
      'canvas': 'canvas'
    }
  }
}