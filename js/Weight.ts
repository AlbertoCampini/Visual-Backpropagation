import {Neuron} from "./Neuron";

export class Weight{

    private _id: number;
    private _to: Neuron;
    private _from: Neuron;
    private _weight: number;
    private _label: string;
    private _delta: number;

    constructor(id: number, to: Neuron, from: Neuron, weight: number) {
        this._id = id;
        this._to = to;
        this._from = from;
        this._weight = weight;
        this._label = weight.toString()
    }


    get delta(): number {
        return this._delta;
    }

    set delta(value: number) {
        this._delta = value;
    }

    get id(): number {
        return this._id;
    }

    set id(value: number) {
        this._id = value;
    }

    get label(): string {
        return this._label;
    }

    set label(value: string) {
        this._label = value;
    }

    get to(): Neuron {
        return this._to;
    }

    set to(value: Neuron) {
        this._to = value;
    }

    get from(): Neuron {
        return this._from;
    }

    set from(value: Neuron) {
        this._from = value;
    }

    get weight(): number {
        return this._weight;
    }

    set weight(value: number) {
        this._weight = value;
    }
}