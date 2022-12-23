import {Neuron} from "./Neuron";

export class Input extends Neuron{
    private _input: number;

    constructor(id: number, layer: number, input: number) {
        super(id, layer);
        this._input = input;
    }

    set input(value: number) {
        this._input = value;
    }

    getOutput(): number {
        return this._input;
    }
}