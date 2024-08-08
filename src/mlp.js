const Layer = require('./layer');

class MLP {
  constructor(nLayers, numNeurons, numInputsPerNeuron, nOutputs) {
    this.layers = [];
    // Camadas intermediárias
    for (let i = 0; i < nLayers - 1; i++) {
      this.layers.push(new Layer(numNeurons, numInputsPerNeuron));
      numInputsPerNeuron = numNeurons; // O número de inputs da próxima camada é o número de neurônios da camada atual
    }
    // Última camada de saída
    this.layers.push(new Layer(nOutputs, numNeurons));
  }

  // Função sigmoid com correção para overflow/underflow
  sigmoid(x) {
    const value = Math.max(-20, Math.min(20, x)); // Ajuste para evitar valores extremos
    return 1 / (1 + Math.exp(-value));
  }

  // Derivada da função sigmoid
  sigmoidDerivative(sigmoidOutput) {
    return sigmoidOutput * (1 - sigmoidOutput);
  }

  feedForward(inputs) {
    let outputs = inputs;
    for (let layer of this.layers) {
      outputs = layer.feedForward(outputs).map(x => this.sigmoid(x));
    }
    return outputs;
  }

  train(data, labels, learningRate, epochs) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalError = 0;

      data.forEach((inputs, index) => {
        let target = labels[index];

        // Feedforward
        let outputs = [inputs];

        this.layers.forEach(layer => {
          let layerOutput = layer.feedForward(outputs[outputs.length - 1]).map(x => this.sigmoid(x));
          outputs.push(layerOutput);
        });

        // Saída final da rede
        let finalOutput = outputs[outputs.length - 1];

        // Backpropagation
        let errors = target.map((t, i) => t - finalOutput[i]);

        totalError += errors.reduce((acc, err) => acc + Math.pow(err, 2), 0);

        for (let i = this.layers.length - 1; i >= 0; i--) {
          let layer = this.layers[i];
          let output = outputs[i + 1];
          let input = outputs[i];

          let gradients = output.map((out, j) => this.sigmoidDerivative(out) * errors[j] * learningRate);

          let deltas = gradients.map(grad => input.map(inp => inp * grad));

          let nextErrors = layer.neurons.map((neuron, k) => {

            return neuron.weights.reduce((sum, weight, j) => sum + weight * gradients[k], 0);
          });

          //console.log(nextErrors)

          layer.neurons.forEach((neuron, k) => {
            neuron.weights = neuron.weights.map((weight, j) => weight + deltas[k][j]);
            neuron.bias += gradients[k];
          });

          // Atualiza os erros para a próxima iteração
          errors = nextErrors;
        }
      });

      // Calculando o erro médio quadrático (MSE)
      totalError /= data.length;
      //console.log(`Epoch ${epoch + 1}, Error: ${totalError}`);
    }
  }
}

module.exports = MLP;
