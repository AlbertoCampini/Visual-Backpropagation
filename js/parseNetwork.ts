import {NeuralNetwork} from "./NeuralNetwork";
import {Input} from "./Input";
import {Neuron} from "./Neuron";
import data from "./data.json";
import {Weight} from "./Weight";
import * as cons from "./costants";
import {forEach} from "mathjs";
import {Bias} from "./Bias";

export
function parseNetwork(input): NeuralNetwork {
    let number_of_unit = input.NeuralNetwork.input_number + input.NeuralNetwork.neuron_number.reduce((a, b) => a + b, 0)
    let id_neuron = 1
    let number_of_weight = input.NeuralNetwork.input_number * input.NeuralNetwork.neuron_number[0] + input.NeuralNetwork.neuron_number.reduce((a, b) => a * b, 1)
    let id_weight = 1

    let input_layer: Input[] = []
    let bias_layer: Bias[] = []
    let internal_layer: Neuron[][] = []

    for (let i = 0; i < input.NeuralNetwork.input_number; i++) {
        input_layer.push(new Input(id_neuron, -1, input.DataSet.data[0].input[i]))
        id_neuron++
    }

    for (let i = 0; i < input.NeuralNetwork.bias.length; i++) {
        bias_layer.push(new Bias(id_neuron, -1, 1))
        id_neuron++
    }

    for (let i = 0; i < input.NeuralNetwork.layer_number; i++) {
        let current_layer = []
        for (let j = 0; j < input.NeuralNetwork.neuron_number[i]; j++) {
            current_layer.push(new Neuron(id_neuron, i,data.NeuralNetwork.activation_function))
            id_neuron++
        }
        internal_layer.push(current_layer)
    }

    for (let i = 0; i < input.NeuralNetwork.layer_number; i++) {
        for (let j = 0; j < input.NeuralNetwork.neuron_number[i]; j++) {
            let current_weights = []
            if (i === 0) {
                input_layer.forEach((current_input,index) => {
                    current_weights.push(new Weight(id_weight, internal_layer[i][j], current_input, cons.getRandomArbitrary()))
                    id_weight++
                    current_weights.push(new Weight(id_weight, internal_layer[i][j], bias_layer[index], cons.getRandomArbitrary()))
                    id_weight++
                })
            } else {
                internal_layer[i - 1].forEach((neuron,index) => {
                    current_weights.push(new Weight(id_weight, internal_layer[i][j], neuron, cons.getRandomArbitrary()))
                    id_weight++
                    current_weights.push(new Weight(id_weight, internal_layer[i][j], bias_layer[index], cons.getRandomArbitrary()))
                    id_weight++
                })
            }
            internal_layer[i][j].input_weights = current_weights
        }
    }



    if (number_of_unit === id_neuron && number_of_weight === id_weight) {
        console.log("success")
    }
    return new NeuralNetwork(internal_layer, input_layer, 0.15)
}