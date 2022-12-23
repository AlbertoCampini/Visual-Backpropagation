import {Neuron} from "./Neuron";

export class Weight{

    id: number;
    to: Neuron;
    from: Neuron;
    weight: number;

    constructor(id: number, to: Neuron, from: Neuron, weight: number) {
        this.id = id;
        this.to = to;
        this.from = from;
        this.weight = weight;
    }
}