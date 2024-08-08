class Neuron {
  constructor(numInputs) {
    this.weights = Array(numInputs).fill(0).map(() => Math.random() * 0.1); // Ajustado para inicialização entre 0 e 0.1
    this.bias = Math.random() * 0.1; // Ajustado para inicialização entre 0 e 0.1
  }

  activate(inputs) {
    let sum = this.weights.reduce((acc, weight, index) => acc + weight * inputs[index], 0) + this.bias;
    return sum; // Retorna a soma sem aplicar a função sigmoid, isso é feito no feedForward da MLP
  }
}

module.exports = Neuron;
