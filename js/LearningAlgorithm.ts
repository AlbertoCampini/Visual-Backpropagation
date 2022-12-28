import {Neuron} from "./Neuron";
import {Weight} from "./Weight";
import {Input} from "./Input";
import {NeuralNetwork} from "./NeuralNetwork";
import data from './data.json'
import math = require("mathjs");
import * as cons from './costants';
import {fixNumber, getRandomArbitrary} from "./costants";
import {Dataset} from "./Dataset";
import {parseDataset} from "./parseDataset";
import {parseNetwork} from "./parseNetwork";

class LearningAlgorithm {
    static backPropagation(neuralNetwork: NeuralNetwork, dataset: Dataset): NeuralNetwork {
        dataset.input.forEach((dataset_input, dataset_index) => {
            neuralNetwork.input = dataset_input
            for (let i = neuralNetwork.visible_layer_level; i >= 0; i--) {
                if (i === neuralNetwork.visible_layer_level) {
                    neuralNetwork.network[i].forEach((neuron, index) => {
                        neuron.signal_error = dataset.target[dataset_index][index] - neuron.getOutput()
                        neuron.input_weights.forEach((weight) => {
                            this.updateWeightBP(weight, neuralNetwork.learning_rate, neuron.getDerivative(), neuron.signal_error, weight.from.getOutput())
                        })
                    })
                } else {
                    neuralNetwork.network[i].forEach((neuron) => {
                        let signal_error = 0
                        neuralNetwork.getConnectedNeuron(neuron.id).forEach((connectedNeuron) => {
                            //console.log("il segnale di errore del neurone: ",connectedNeuron.id," è: ",connectedNeuron.signal_error)
                            signal_error = signal_error + (connectedNeuron.signal_error * neuralNetwork.getWeightByConnection(connectedNeuron.id, neuron).weight)
                        })
                        neuron.signal_error = signal_error
                        neuron.input_weights.forEach((weight) => {
                            this.updateWeightBP(weight, neuralNetwork.learning_rate, neuron.getDerivative(), neuron.signal_error, weight.from.getOutput())

                        })
                    })
                }

            }
        })
        return neuralNetwork
    }

    static generalizedDeltaRule(neuralNetwork: NeuralNetwork, dataset: Dataset): NeuralNetwork {
        dataset.input.forEach((dataset_input, dataset_index) => {
            neuralNetwork.input = dataset_input
            for (let i = neuralNetwork.visible_layer_level; i >= 0; i--) {
                if (i === neuralNetwork.visible_layer_level) {
                    neuralNetwork.network[i].forEach((neuron, index) => {
                        neuron.signal_error = dataset.target[dataset_index][index] - neuron.getOutput()
                        neuron.input_weights.forEach((weight) => {
                            this.updateWeightGDR(weight, 0.2, neuralNetwork.learning_rate, neuron.getDerivative(), neuron.signal_error, weight.from.getOutput())
                        })
                    })
                } else {
                    neuralNetwork.network[i].forEach((neuron) => {
                        let signal_error = 0
                        neuralNetwork.getConnectedNeuron(neuron.id).forEach((connectedNeuron) => {
                            signal_error = signal_error + (connectedNeuron.signal_error * neuralNetwork.getWeightByConnection(connectedNeuron.id, neuron).weight)
                        })
                        neuron.signal_error = signal_error
                        neuron.input_weights.forEach((weight) => {
                            this.updateWeightGDR(weight, 0.2, neuralNetwork.learning_rate, neuron.getDerivative(), neuron.signal_error, weight.from.getOutput())

                        })
                    })
                }

            }
        })
        return neuralNetwork
    }

    private static updateWeightGDR(weight: Weight, alpha: number, learning_rate: number, derivative: number, signal_error: number, output: number) {
        let last_delta = 0
        weight.label = fixNumber(weight.weight, 3).toString() + " + (" + nn.learning_rate + " * " + fixNumber(derivative, 3) + " * " + fixNumber(signal_error, 3) + " * " + fixNumber(output, 3) + ")"
        last_delta = weight.delta
        weight.delta = alpha * last_delta * learning_rate * derivative * signal_error * output
        weight.weight = weight.weight + weight.delta
    }

    private static updateWeightBP(weight: Weight, learning_rate: number, derivative: number, signal_error: number, output: number) {
        weight.label = fixNumber(weight.weight, 3).toString() + " + (" + nn.learning_rate + " * " + fixNumber(derivative, 3) + " * " + fixNumber(signal_error, 3) + " * " + fixNumber(output, 3) + ")"
        weight.delta = learning_rate * derivative * signal_error * output
        weight.weight = weight.weight + weight.delta
    }
}


/*ES 1
let input_1 = new Input(1, -1, -0.5)
let input_2 = new Input(2, -1, 0.37)
let input_3 = new Input(3, -1, 1)//Bias


let visible_neuron = new Neuron(6, 1, "relu")


let input_layer_visible = [new Weight(1, visible_neuron, input_1, 0.2), new Weight(2, visible_neuron, input_2, 0.33), new Weight(2, visible_neuron, input_3, 0.1)]

visible_neuron.input_weights = input_layer_visible

let layer_0 = [input_1, input_2, input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1], layer_0, 0.25)
let dt = new Dataset([[-0.5, 0.37, 1]], [[0.5]])
*/
//ES esame
let input_1 = new Input(1, -1, 0.4)
let input_2 = new Input(2, -1, 0.2)
let input_3 = new Input(3, -1, 1)//Bias


let visible_neuron = new Neuron(6, 1, "relu")


let input_layer_visible = [new Weight(1, visible_neuron, input_1, -0.3), new Weight(2, visible_neuron, input_2, 0.6), new Weight(2, visible_neuron, input_3, 0.2)]

visible_neuron.input_weights = input_layer_visible

let layer_0 = [input_1, input_2, input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1], layer_0, 0.15)
let dt = new Dataset([[0.4, 0.2, 1]], [[0.4]])
/*
// ES 2
let input_1 = new Input(1, -1, 0.4)
let input_2 = new Input(2, -1, 0.2)
let input_3 = new Input(3, -1, 1)//Bias


let visible_neuron = new Neuron(6, 1)


let input_layer_visible = [new Weight(1, visible_neuron, input_1, -0.3), new Weight(2, visible_neuron, input_2, 0.6), new Weight(2, visible_neuron, input_3, 0.2)]

let layer_0 = [input_1, input_2, input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1], layer_0,0.15)

visible_neuron.input_weights=input_layer_visible
let dt = new Dataset([[0.4,0.2,1]],[[0.4]])



let input_1 = new Input(1,-1,1)

let hidden_neuron_1 = new Neuron(2,0,)
let hidden_neuron_2 = new Neuron(3,0,)
let hidden_neuron_3 = new Neuron(4,0,)//Bias


let visible_neuron_1 = new Neuron(6,1)
let visible_neuron_2 = new Neuron(7,1)

let input_layer_hidden_1 = [new Weight(1,input_1,hidden_neuron_1,-3)]
let input_layer_hidden_2 = [new Weight(2,input_1,hidden_neuron_2,6)]
let input_layer_hidden_3 = [new Weight(3,input_1,hidden_neuron_3,2)]

let input_layer_visible_layer_1= [new Weight(4,hidden_neuron_1,visible_neuron_1,-3),new Weight(5,hidden_neuron_2,visible_neuron_1,6),new Weight(6,hidden_neuron_3,visible_neuron_1,2)]
let input_layer_visible_layer_2 = [new Weight(7,hidden_neuron_1,visible_neuron_2,-3),new Weight(8,hidden_neuron_2,visible_neuron_2,6),new Weight(9,hidden_neuron_3,visible_neuron_2,2)]


let input = [input_1]
let layer_0 = [hidden_neuron_1,hidden_neuron_2,hidden_neuron_3]
let layer_1 = [visible_neuron_1,visible_neuron_2]

let nn = new NeuralNetwork([layer_0,layer_1],[input])

hidden_neuron_1.input_weights(input_layer_hidden_1)
hidden_neuron_2.input_weights(input_layer_hidden_2)
hidden_neuron_3.input_weights(input_layer_hidden_3)

visible_neuron_1.input_weights(input_layer_visible_layer_1)
visible_neuron_2.input_weights(input_layer_visible_layer_2)

*/

//console.log(nn.getConnectedNeuron(2))

nn = parseNetwork(data)
dt = parseDataset(data)
console.log(dt)
//console.log(nn.weights)
console.log(nn.evaluate(dt))

let error = 1
let max = 1000000
while (error != 0 && max != 0) {
    error = 0
    nn = LearningAlgorithm.backPropagation(nn, dt)
    let res = nn.evaluate(dt)
    res.forEach((res) => {
        error = error + res.target - (res.output >= 0.5 ? 1 : 0)
    })
    max--
}


//console.log(nn.weights)
//nn.updateWeights()
nn = LearningAlgorithm.backPropagation(nn, dt)
nn.network.forEach((level) => {
    level.forEach((neuron) => {
        neuron.input_weights.forEach((w) => {
            console.log(neuron.id, ": to ", w.to.id, ", from ", w.from.id, w.weight, w.label)
        })
    })
})


console.log(nn.evaluate(dt))


//console.log(nn.weights)
//console.log(nn.network[1][0]._input_weights)

//console.log(nn.network[nn.visible_layer_level][0]._input_weights)



