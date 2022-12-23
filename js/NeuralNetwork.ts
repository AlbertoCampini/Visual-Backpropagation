import {Neuron} from "./Neuron";
import {Input} from "./Input";

export class NeuralNetwork{
    input: Input[][]
    network: Neuron[][]
    learning_rate = 0.15
    visible_layer_level: number

    constructor(network: Neuron[][],input: Input[][]) {
        this.network = network;
        this.input = input;
        this.visible_layer_level =  network.length - 1;

    }

    evaluate(): number[]{
        let result = [];
        this.network[this.visible_layer_level].forEach((output_neuron) =>{
            result.push(output_neuron.getOutput())
        })
        return result;
    }
}