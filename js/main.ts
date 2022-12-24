import {Neuron} from "./Neuron";
import {Weight} from "./Weight";
import {Input} from "./Input";
import {NeuralNetwork} from "./NeuralNetwork";
import math = require("mathjs");
import vis = require("vis");

/*
let input_1 = new Input(1,-1,1) //Bias
let input_2 = new Input(2,-1,0)
let input_3 = new Input(3,-1,0)

let hidden_neuron_1 =new Neuron(4,0)
let hidden_neuron_2 =new Neuron(5,0)

let visible_neuron = new Neuron(6,1)

let input_hidden_neuron_1 = [new Weight(1,input_1,hidden_neuron_1,0.1),new Weight(1,input_2,hidden_neuron_1,0.4),new Weight(1,input_3,hidden_neuron_1,-0.2)]
let input_hidden_neuron_2 = [new Weight(2,input_1,hidden_neuron_2,-0.1),new Weight(2,input_2,hidden_neuron_2,-0.3),new Weight(2,input_3,hidden_neuron_2,0.5)]
let hidden_layer_visible = [new Weight(1,hidden_neuron_1,visible_neuron,1),new Weight(2,hidden_neuron_2,visible_neuron,-1)]

let layer_0 = [input_1,input_2,input_3]
let layer_1 = [hidden_neuron_1,hidden_neuron_2]
let layer_2 = [visible_neuron]

let nn = new NeuralNetwork([layer_1, layer_2],[layer_0])

hidden_neuron_1.input_weights(input_hidden_neuron_1)
hidden_neuron_2.input_weights(input_hidden_neuron_2)
visible_neuron.input_weights(hidden_layer_visible)
*/
console.log(math.derivative("x^2","x"))
