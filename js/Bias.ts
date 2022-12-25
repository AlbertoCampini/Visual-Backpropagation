import {Neuron} from "./Neuron";

export class Bias extends Neuron{
    private _bias: number;

    constructor(id: number, layer: number, bias: number) {
        super(id, layer);
        this._bias = bias;
    }

    set input(value: number) {
        this._bias = value;
    }

    get input(): number {
        return this._bias;
    }

    getOutput(): number {
        return this._bias;
    }
}