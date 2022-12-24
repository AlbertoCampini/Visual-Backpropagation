import {Neuron} from "./Neuron";
import {Input} from "./Input";
import {Weight} from "./Weight";
import {neutronMassDependencies} from "mathjs";

export class NeuralNetwork{
    input: Input[]
    private readonly _network: Neuron[][]
    private _weights: Weight[] = []
    private readonly _learning_rate: number
    private _visible_layer_level: number

    constructor(network: Neuron[][],input: Input[], learning_rate: number = 1) {
        this._network = network;
        this.input = input;
        this._learning_rate = learning_rate
        this._visible_layer_level =  network.length - 1;
        for (let i = 0; i < network.length; i++) {
            for (let j = 0; j < network[i].length; j++) {
                network[i][j]._input_weights.forEach((weight) =>{
                    this._weights.push(weight)
                })
            }
        }
    }

    get visible_layer_level(): number {
        return this._visible_layer_level;
    }

    get network(): Neuron[][] {
        return this._network;
    }

    get learning_rate(): number {
        return this._learning_rate;
    }

    get weights(): Weight[] {
        return this._weights;
    }

    updateWeights(){
        for (let i = 0; i < this._network.length; i++) {
            for (let j = 0; j < this._network[i].length; j++) {
                this._network[i][j]._input_weights.forEach((weight) =>{
                    this._weights.push(weight)
                })
            }
        }
    }

    getNeuron(id: number): Neuron{
        let response = null
        for (let i = 0; i <= this._visible_layer_level && response === null; i++) {
            this._network[i].forEach((neuron)=>{
                if(neuron.id === id)
                    response = neuron
            })
        }
        return response
    }

    getWeightByID(id: number): Weight{
        let response = null
        for (let i = 0; i < this._visible_layer_level; i++) {
            this._network[i].forEach((neuron)=>{
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
        if(current_neuron.layer+1 > this._visible_layer_level)
            return null
        this._network[current_neuron.layer+1].forEach((neuron) =>{
            neuron._input_weights.forEach((weight)=>{
                if(weight.to === current_neuron)
                    connected_neuron.push(weight.from)
            })
        })
        return connected_neuron
    }

    evaluate(): number[]{
        let result = [];
        this._network[this._visible_layer_level].forEach((output_neuron) =>{
            result.push(output_neuron.getOutput())
        })
        return result;
    }
}