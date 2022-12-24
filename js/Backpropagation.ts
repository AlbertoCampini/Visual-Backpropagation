import {Neuron} from "./Neuron";
import {Weight} from "./Weight";
import {Input} from "./Input";
import {NeuralNetwork} from "./NeuralNetwork";
import data from './data.json'
import math = require("mathjs");
import * as cons from './costants';
import {getRandomArbitrary} from "./costants";


let d = [0.4]

//FORWARD PASS
function forwardStep(nn: NeuralNetwork): number[] {
    return nn.evaluate();
}

function backPropagation(nn: NeuralNetwork, target: number[]): NeuralNetwork {
    for (let i = nn.visible_layer_level; i >= 0; i--) {
        if (i === nn.visible_layer_level) {
            nn.network[i].forEach((neuron) => {
                neuron.signal_error(target[0] - neuron.getOutput())
                neuron._input_weights.forEach((weight) => {
                    //console.log("Delta W per il peso che connette: ",weight.to.id," ",weight.from.id,": ","*",nn.learning_rate,"*",neuron._signal_error,"*",neuron.getDerivative(),"*",weight.to.getOutput())
                    weight.weight = weight.weight + nn.learning_rate * neuron._signal_error * neuron.getDerivative() * weight.to.getOutput()
                })
            })
        } else {
            let signal_error = 0
            nn.network[i].forEach((neuron) => {
                nn.getConnectedNeuron(neuron.id).forEach((connectedNeuron) => {
                    signal_error = signal_error + connectedNeuron._signal_error * nn.getWeightByConnection(connectedNeuron.id, neuron).weight * neuron.getDerivative()
                })

                neuron._input_weights.forEach((weight) => {
                    //console.log("Delta W per il peso che connette: ",weight.to.id," ",weight.from.id,": ",nn.learning_rate,"*",signal_error,"*",weight.to.getOutput())
                    weight.weight = weight.weight + nn.learning_rate * signal_error * weight.to.getOutput()

                })
            })
        }

    }

    return nn
}

/*
//ES 1
let input_1 = new Input(1,-1,-0.5)
let input_2 = new Input(2,-1,0.37)
let input_3 = new Input(3,-1,1)//Bias


let visible_neuron = new Neuron(6,1)


let input_layer_visible = [new Weight(1,input_1,visible_neuron,0.2),new Weight(2,input_2,visible_neuron,0.33),new Weight(2,input_3,visible_neuron,0.1)]

let layer_0 = [input_1,input_2,input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1],[layer_0])

*/

// ES 2
let input_1 = new Input(1, -1, 0.4)
let input_2 = new Input(2, -1, 0.2)
let input_3 = new Input(3, -1, 1)//Bias


let visible_neuron = new Neuron(6, 1)


let input_layer_visible = [new Weight(1, visible_neuron, input_1, -0.3), new Weight(2, visible_neuron, input_2, 0.6), new Weight(2, visible_neuron, input_3, 0.2)]

let layer_0 = [input_1, input_2, input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1], layer_0)

visible_neuron.input_weights(input_layer_visible)
/*


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


function parseNetwork(input): NeuralNetwork {
    let number_of_unit = input.NeuralNetwork.input_number + input.NeuralNetwork.neuron_number.reduce((a, b) => a + b, 0)
    let id_neuron = 1
    let number_of_weight = input.NeuralNetwork.input_number * input.NeuralNetwork.neuron_number[0] + input.NeuralNetwork.neuron_number.reduce((a, b) => a * b, 1)
    let id_weight = 1

    let input_layer: Input[] = []
    let internal_layer: Neuron[][] = []

    for (let i = 0; i < input.NeuralNetwork.input_number; i++) {
        input_layer.push(new Input(id_neuron, -1, input.TrainingSet.data[i].input))
        id_neuron++
    }

    for (let i = 0; i < input.NeuralNetwork.layer_number; i++) {
        let current_layer = []
        for (let j = 0; j < input.NeuralNetwork.neuron_number[i]; j++) {
            current_layer.push(new Neuron(id_neuron, i))
            id_neuron++
        }
        internal_layer.push(current_layer)
    }

    for (let i = 0; i < input.NeuralNetwork.layer_number; i++) {
        for (let j = 0; j < input.NeuralNetwork.neuron_number[i]; j++) {
            let current_weights = []
            if (i === 0) {
                input_layer.forEach((input) => {
                    current_weights.push(new Weight(id_weight, internal_layer[i][j], input, cons.getRandomArbitrary()))
                    id_weight++
                })
            } else {
                internal_layer[i - 1].forEach((neuron) => {
                    current_weights.push(new Weight(id_weight, internal_layer[i][j], neuron, cons.getRandomArbitrary()))
                    id_weight++
                })
            }
            internal_layer[i][j].input_weights(current_weights)
        }
    }

    if (number_of_unit === id_neuron && number_of_weight === id_weight) {
        console.log("success")
    }
    return new NeuralNetwork(internal_layer, input_layer, 0.15)
}

nn = parseNetwork(data)
//console.log(nn.weights)
console.log(nn.evaluate())


let weight = []
for (let i = 0; i < 20000; i++) {
    nn = backPropagation(nn, d)
    nn.updateWeights()
    weight.push(nn.weights)
}
console.log(nn.evaluate())

//console.log(nn.network[nn.visible_layer_level][0]._input_weights)



