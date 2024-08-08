const fs = require('fs');
const csv = require('csv-parser');
const MLP = require('./src/mlp');

const filePath = 'dataset/IRIS.csv';

const features = [];
const labels = [];

fs.createReadStream(filePath)
  .pipe(csv())
  .on('data', (row) => {
    features.push([
      parseFloat(row.sepal_length),
      parseFloat(row.sepal_width),
      parseFloat(row.petal_length),
      parseFloat(row.petal_width)
    ]);

    labels.push(row.species);
  })
  .on('end', () => {
    const encodedLabels = labels.map(label => {
      if (label === 'Iris-setosa') return [1, 0, 0];
      else if (label === 'Iris-versicolor') return [0, 1, 0];
      else if (label === 'Iris-virginica') return [0, 0, 1];
    });

    const splitIndex = Math.floor(features.length * 0.8);
    const trainFeatures = features.slice(0, splitIndex);
    const trainLabels = encodedLabels.slice(0, splitIndex);
    const testFeatures = features.slice(splitIndex);
    const testLabels = encodedLabels.slice(splitIndex);

    const mlp = new MLP([2, 4, 4, 3]); // nLayers, numNeurons, numInputsPerNeuron, nOutputs

    mlp.train(trainFeatures, trainLabels, 0.1, 10000); // Treinamento com taxa de aprendizado de 0.1 por 10000 épocas

    let correct = 0;

    testFeatures.forEach((inputs, index) => {
      const output = mlp.feedForward(inputs);
      correct = correct + output[0]
    });

    console.log("% de acerto: ", ( 100 - (correct/testFeatures.length) * 100))
  })
  .on('error', (error) => {
    console.error('Error:', error);
  });
