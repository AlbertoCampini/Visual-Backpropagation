import {Weight} from './Weight';
import * as cons from './costants';

export class Neuron {
    private _id: number;
    private _layer: number;
    private readonly _activation_function: string;
    private _signal_error: number;
    private _input_weights: Weight[] = []

    constructor(id: number, layer: number, activation_function: string = cons.ACTIVATION_FUNCTION) {
        this._activation_function = activation_function === null ? cons.ACTIVATION_FUNCTION : activation_function
        this._id = id;
        this._layer = layer;
    }


    get id(): number {
        return this._id;
    }

    set id(value: number) {
        this._id = value;
    }

    get layer(): number {
        return this._layer;
    }

    set layer(value: number) {
        this._layer = value;
    }

    set signal_error(value: number) {
        this._signal_error = value;
    }

    get signal_error(): number {
        return this._signal_error;
    }

    get input_weights(): Weight[] {
        return this._input_weights;
    }

    set input_weights(value: Weight[]) {
        this._input_weights = value;
    }

    /**
     * Evaluate the input for neuron v_j, this is the weighted sum of weigh and output of connected neuron
     * @private
     */
    private getInput(): number {
        let output = 0;
        this._input_weights.forEach((w) => {
            output = output + w.weight * w.from.getOutput();
        })
        return output;
    }

    /**
     * Evaluate the output y_j, this is the result of activation function application on neuron input v_j
     */
    getOutput(): number {
        return this.activationFunction()
    }

    getDerivative(): number{
        switch (this._activation_function){
            case "relu":
                return this.getInput() > 0 ? 1 : 0
            case "sigmoid":
                //console.log("la derivata prima di: ",this.getInput()," è ",Math.pow(Math.E,this.getInput()*-1)/Math.pow(1+Math.pow(Math.E,this.getInput()*-1),2))
                return Math.pow(Math.E,this.getInput()*-1)/Math.pow(1+Math.pow(Math.E,this.getInput()*-1),2)
        }

    }

    private activationFunction(): number {
        switch (this._activation_function){
            case "relu":
                return this.getInput() > 0 ? this.getInput() : 0
            case "sigmoid":
                //console.log("l'input al neurone ",this.id, " è: ",this.getInput(), " e con output: ",1 / (1 + Math.pow(Math.E, -this.getInput())))
                return 1 / (1 + Math.pow(Math.E, -this.getInput()))
        }
    }
}