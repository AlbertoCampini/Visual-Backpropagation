import {Weight} from './Weight';

export class Neuron {
    id: number;
    layer: number;
    _signal_error: number;
    _input_weights: Weight[]

    constructor(id: number, layer: number) {
        this.id = id;
        this.layer = layer;
    }

    signal_error(value: number) {
        this._signal_error = value;
    }

    input_weights(value: Weight[]) {
        this._input_weights = value;
    }

    private getInput(): number {
        let output = 0;
        this._input_weights.forEach((w) => {
            output = output + w.weight * w.to.getOutput();
        })
        return output;
    }

    getOutput(): number {
        return this.activationFunction(this.getInput())
    }

    getDerivative(): number{
        //return 1
        return Math.pow(Math.E,this.getOutput()*-1)/Math.pow(1+Math.pow(Math.E,this.getOutput()*-1),2)
    }

    private activationFunction(input): number {
        return 1 / (1 + Math.pow(Math.E, -input)) //Sigmoide
        //return input > 0 ? 1 : -1 //
        //return input
    }
}