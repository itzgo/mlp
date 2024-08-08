const Neuron = require('./neuron');

class Layer {
  constructor(numNeurons, numInputsPerNeuron) {
    this.neurons = Array(numNeurons).fill(0).map(() => new Neuron(numInputsPerNeuron));
  }

  feedForward(inputs) {
    return this.neurons.map(neuron => neuron.activate(inputs));
  }
}

module.exports = Layer;
