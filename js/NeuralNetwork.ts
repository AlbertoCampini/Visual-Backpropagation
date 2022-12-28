import {Neuron} from "./Neuron";
import {Input} from "./Input";
import {Weight} from "./Weight";
import {Dataset} from "./Dataset";
import {neutronMassDependencies} from "mathjs";

export class NeuralNetwork {
    private _input: Input[]
    private readonly _network: Neuron[][]
    private _weights: Weight[] = []
    private readonly _learning_rate: number
    private readonly _visible_layer_level: number

    constructor(network: Neuron[][], input: Input[], learning_rate: number = 1) {
        this._network = network;
        this._input = input;
        this._learning_rate = learning_rate
        this._visible_layer_level = network.length - 1;
        for (let i = 0; i < network.length; i++) {
            for (let j = 0; j < network[i].length; j++) {
                network[i][j].input_weights.forEach((weight) => {
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

    getNeuron(id: number): Neuron {
        let response = null
        for (let i = 0; i <= this._visible_layer_level && response === null; i++) {
            this._network[i].forEach((neuron) => {
                if (neuron.id === id)
                    response = neuron
            })
        }
        return response
    }

    /**
     * Get the Weight identified by id passed
     * @param id
     */
    getWeightByID(id: number): Weight {
        let response = null
        for (let i = 0; i < this._visible_layer_level; i++) {
            this._network[i].forEach((neuron) => {
                neuron.input_weights.forEach((weight) => {
                    if (weight.id === id)
                        response = weight
                })
            })
        }
        return response
    }

    /**
     * Get the Weight that connect the neuron identified by id passed and neuron passed
     * @param id
     * @param neuron
     */
    getWeightByConnection(id: number, neuron: Neuron): Weight {
        let response = null
        let current_neuron = this.getNeuron(id)
        if (current_neuron === null)
            return null
        current_neuron.input_weights.forEach((weight) => {
            if (weight.from === neuron)
                response = weight
        })

        return response
    }

    /**
     * Return a list of neuron connect with neuron identified by id passed
     * @param id
     */
    getConnectedNeuron(id: number): Neuron[] {
        let current_neuron = this.getNeuron(id)
        if (current_neuron === null)
            return null

        let connected_neuron: Neuron[] = []
        if (current_neuron.layer + 1 > this._visible_layer_level)
            return null
        this._network[current_neuron.layer + 1].forEach((neuron) => {
            neuron.input_weights.forEach((weight) => {
                if (weight.from === current_neuron)
                    connected_neuron.push(weight.to)
            })
        })
        return connected_neuron
    }

    set input(value: any[]) {
        this._input.forEach((network_input, index) => {
            network_input.input = value[index]
        })
    }

    /**
     * Return the Result of Neural Network on each example contained into the passed dataset
     * @param dataset
     */
    evaluate(dataset: Dataset = null){
        let result = [];
        if (dataset === null) {
            this._network[this._visible_layer_level].forEach((output_neuron) => {
                result[0].output.push(output_neuron.getOutput())
            })
            this._input.forEach((input) => {
                result[0].input.push(input.input)
            })
        } else {
            dataset.input.forEach((dataset_input, result_index) => {
                result[result_index] = {
                    input: [],
                    output: [],
                    target: []
                }
                this._input.forEach((network_input, index) => {
                    network_input.input = dataset_input[index] //prensento gli input alla rete
                    result[result_index].input.push(dataset_input[index]) //inserisco nell'oggetto result il valore degli input
                })
                this._network[this._visible_layer_level].forEach((output_neuron, index) => {
                    result[result_index].output.push(output_neuron.getOutput()) //salvo l'output ottenuto con la configuarazione di pesi corrente
                    result[result_index].target.push(dataset.target[result_index][index]) //salvo il target che avrei voluto ricevere con quell'input
                })

            })

        }

        return result;
    }
}