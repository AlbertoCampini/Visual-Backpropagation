import {Weight} from './Weight';
import * as cons from './costants';

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
            //console.log("sono il neurone:",this.id," ",output," + ",w.weight,"*",w.to.getOutput())
            output = output + w.weight * w.to.getOutput();
        })
        return output;
    }

    getOutput(): number {
        return this.activationFunction()
    }

    getDerivative(): number{
        switch (cons.ACTIVATION_FUNCTION){
            case "relu":
                return this.getInput() > 0 ? 1 : 0
            case "sigmoid":
                //console.log("la derivata prima di: ",this.getInput()," è ",Math.pow(Math.E,this.getInput()*-1)/Math.pow(1+Math.pow(Math.E,this.getInput()*-1),2))
                return Math.pow(Math.E,this.getInput()*-1)/Math.pow(1+Math.pow(Math.E,this.getInput()*-1),2)
        }

    }

    private activationFunction(): number {
        switch (cons.ACTIVATION_FUNCTION){
            case "relu":
                return this.getInput() > 0 ? 1 : 0
            case "sigmoid":
                //console.log("l'input al neurone ",this.id, " è: ",this.getInput(), " e con output: ",1 / (1 + Math.pow(Math.E, -this.getInput())))
                return 1 / (1 + Math.pow(Math.E, -this.getInput()))
        }
    }
}