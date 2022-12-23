import {Neuron} from "./Neuron";
import {Weight} from "./Weight";
import {Input} from "./Input";
import {NeuralNetwork} from "./NeuralNetwork";
import math = require("mathjs");


let d = [0.4]
//FORWARD PASS
function forwardStep(nn: NeuralNetwork): number[]{
    return nn.evaluate();
}
let delta_w = []
function backPropagation(nn: NeuralNetwork, target: number[]): NeuralNetwork{
    for (let i = nn.visible_layer_level; i >= 0 ; i--) {
        let delta_w_n = []
        if(i == nn.visible_layer_level){
            nn.network[i].forEach((neuron)=>{
                neuron.signal_error(target[0] - neuron.getOutput())
                neuron._input_weights.forEach((weight)=>{
                    let delta = nn.learning_rate*neuron._signal_error*neuron.getDerivative()*weight.to.getOutput()
                    weight.weight = weight.weight + delta
                    delta_w_n.push(delta)
                })
            })
        }else{

        }
        delta_w.push(delta_w_n)
    }
    console.log(delta_w)
    return nn
}
/*
//ES 1
let input_1 = new Input(1,-1,-0.5)
let input_2 = new Input(2,-1,0.37)
let input_3 = new Input(3,-1,1)//Bias


let visible_neuron = new Neuron(6,1)


let input_layer_visible = [new Weight(1,input_1,visible_neuron,0.2),new Weight(2,input_2,visible_neuron,0.33),new Weight(2,input_3,visible_neuron,0.1)]

let layer_0 = [input_1,input_2,input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1],[layer_0])

*/
// ES 2
let input_1 = new Input(1,-1,0.4)
let input_2 = new Input(2,-1,0.2)
let input_3 = new Input(3,-1,1)//Bias


let visible_neuron = new Neuron(6,1)


let input_layer_visible = [new Weight(1,input_1,visible_neuron,-0.3),new Weight(2,input_2,visible_neuron,0.6),new Weight(2,input_3,visible_neuron,0.2)]

let layer_0 = [input_1,input_2,input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1],[layer_0])

visible_neuron.input_weights(input_layer_visible)
console.log(nn.evaluate())
nn = backPropagation(nn,d)

/*
for (let i = 0; i < 100; i++) {
    nn = backPropagation(nn,d)
    console.log(nn.evaluate())
}
*/
console.log(nn.network[nn.visible_layer_level][0]._input_weights)



