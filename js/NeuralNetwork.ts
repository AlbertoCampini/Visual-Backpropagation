import {Neuron} from "./Neuron";
import {Input} from "./Input";
import {Weight} from "./Weight";

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

    getNeuron(id: number): Neuron{
        let response = null
        for (let i = 0; i <= this.visible_layer_level && response === null; i++) {
            this.network[i].forEach((neuron)=>{
                if(neuron.id === id)
                    response = neuron
            })
        }
        return response
    }

    getWeightByID(id: number): Weight{
        let response = null
        for (let i = 0; i < this.visible_layer_level; i++) {
            this.network[i].forEach((neuron)=>{
                neuron._input_weights.forEach((weight)=>{
                    if(weight.id === id)
                        response = weight
                })
            })
        }
        return response
    }

    getWeightByConnection(id: number, neuron: Neuron): Weight{
        let response = null
        let current_neuron = this.getNeuron(id)
        if(current_neuron === null)
            return null
        current_neuron._input_weights.forEach((weight)=>{
            if(weight.to === neuron)
                response = weight
        })

        return response
    }

    getConnectedNeuron(id: number): Neuron[]{
        let current_neuron = this.getNeuron(id)
        if(current_neuron === null)
            return null

        let connected_neuron: Neuron[] = []
        if(current_neuron.layer+1 > this.visible_layer_level)
            return null
        this.network[current_neuron.layer+1].forEach((neuron) =>{
            neuron._input_weights.forEach((weight)=>{
                if(weight.to === current_neuron)
                    connected_neuron.push(weight.from)
            })
        })
        return connected_neuron
    }

    evaluate(): number[]{
        let result = [];
        this.network[this.visible_layer_level].forEach((output_neuron) =>{
            result.push(output_neuron.getOutput())
        })
        return result;
    }
}