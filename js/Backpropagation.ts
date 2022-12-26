import {Neuron} from "./Neuron";
import {Weight} from "./Weight";
import {Input} from "./Input";
import {NeuralNetwork} from "./NeuralNetwork";
import data from './data.json'
import math = require("mathjs");
import * as cons from './costants';
import {getRandomArbitrary} from "./costants";
import {Dataset} from "./Dataset";
import {parseDataset} from "./parseDataset";
import {parseNetwork} from "./parseNetwork";


let d = [0.4,0.9]

//FORWARD PASS
function forwardStep(nn: NeuralNetwork): {} {
    return nn.evaluate();
}

function backPropagation(nn: NeuralNetwork, dataset: Dataset): NeuralNetwork {
    dataset.input.forEach((dataset_input, dataset_index)=>{
        nn.input = dataset_input
        for (let i = nn.visible_layer_level; i >= 0; i--) {
            if (i === nn.visible_layer_level) {
                nn.network[i].forEach((neuron,index) => {
                    neuron.signal_error = dataset.target[dataset_index][index] - neuron.getOutput()
                    neuron.input_weights.forEach((weight) => {
                        //console.log("Delta W per il peso che connette: ",weight.to.id," ",weight.from.id,": ","*",nn.learning_rate,"*",neuron.signal_error,"*",neuron.getDerivative(),"*",weight.from.getOutput())
                        weight.weight = weight.weight + nn.learning_rate * neuron.signal_error * neuron.getDerivative() * weight.from.getOutput()
                    })
                })
            } else {

                nn.network[i].forEach((neuron) => {
                    let signal_error = 0
                    nn.getConnectedNeuron(neuron.id).forEach((connectedNeuron) => {
                        //console.log("il segnale di errore del neurone: ",connectedNeuron.id," è: ",connectedNeuron.signal_error)
                        signal_error = signal_error + (connectedNeuron.signal_error * nn.getWeightByConnection(connectedNeuron.id, neuron).weight * neuron.getDerivative())
                    })

                    neuron.input_weights.forEach((weight) => {
                        //console.log("Delta W per il peso che connette: ",weight.to.id," ",weight.from.id,": ",nn.learning_rate,"*",signal_error,"*",weight.from.getOutput())
                        weight.weight = weight.weight + nn.learning_rate * signal_error * weight.from.getOutput()

                    })
                })
            }

        }
    })


    return nn
}


//ES 1
let input_1 = new Input(1,-1,-0.5)
let input_2 = new Input(2,-1,0.37)
let input_3 = new Input(3,-1,1)//Bias


let visible_neuron = new Neuron(6,1,"relu")


let input_layer_visible = [new Weight(1,visible_neuron,input_1,0.2),new Weight(2,visible_neuron,input_2,0.33),new Weight(2,visible_neuron,input_3,0.1)]

visible_neuron.input_weights = input_layer_visible

let layer_0 = [input_1,input_2,input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1],layer_0,0.25)
let dt = new Dataset([[-0.5,0.37,1]],[[0.5]])

/*
// ES 2
let input_1 = new Input(1, -1, 0.4)
let input_2 = new Input(2, -1, 0.2)
let input_3 = new Input(3, -1, 1)//Bias


let visible_neuron = new Neuron(6, 1)


let input_layer_visible = [new Weight(1, visible_neuron, input_1, -0.3), new Weight(2, visible_neuron, input_2, 0.6), new Weight(2, visible_neuron, input_3, 0.2)]

let layer_0 = [input_1, input_2, input_3]
let layer_1 = [visible_neuron]

let nn = new NeuralNetwork([layer_1], layer_0,0.15)

visible_neuron.input_weights=input_layer_visible
let dt = new Dataset([[0.4,0.2,1]],[[0.4]])



let input_1 = new Input(1,-1,1)

let hidden_neuron_1 = new Neuron(2,0,)
let hidden_neuron_2 = new Neuron(3,0,)
let hidden_neuron_3 = new Neuron(4,0,)//Bias


let visible_neuron_1 = new Neuron(6,1)
let visible_neuron_2 = new Neuron(7,1)

let input_layer_hidden_1 = [new Weight(1,input_1,hidden_neuron_1,-3)]
let input_layer_hidden_2 = [new Weight(2,input_1,hidden_neuron_2,6)]
let input_layer_hidden_3 = [new Weight(3,input_1,hidden_neuron_3,2)]

let input_layer_visible_layer_1= [new Weight(4,hidden_neuron_1,visible_neuron_1,-3),new Weight(5,hidden_neuron_2,visible_neuron_1,6),new Weight(6,hidden_neuron_3,visible_neuron_1,2)]
let input_layer_visible_layer_2 = [new Weight(7,hidden_neuron_1,visible_neuron_2,-3),new Weight(8,hidden_neuron_2,visible_neuron_2,6),new Weight(9,hidden_neuron_3,visible_neuron_2,2)]


let input = [input_1]
let layer_0 = [hidden_neuron_1,hidden_neuron_2,hidden_neuron_3]
let layer_1 = [visible_neuron_1,visible_neuron_2]

let nn = new NeuralNetwork([layer_0,layer_1],[input])

hidden_neuron_1.input_weights(input_layer_hidden_1)
hidden_neuron_2.input_weights(input_layer_hidden_2)
hidden_neuron_3.input_weights(input_layer_hidden_3)

visible_neuron_1.input_weights(input_layer_visible_layer_1)
visible_neuron_2.input_weights(input_layer_visible_layer_2)

*/

//console.log(nn.getConnectedNeuron(2))

nn = parseNetwork(data)
dt = parseDataset(data)
console.log(dt)
//console.log(nn.weights)
console.log(nn.evaluate(dt))

let error = 1
let max = 1000000
while (error != 0 && max != 0){
    error = 0
    nn = backPropagation(nn, dt)
    let res = nn.evaluate(dt)
    res.forEach((res)=>{
        error = error + res.target - (res.output >= 0.5 ? 1: 0)
    })
max--
}
//console.log(nn.weights)
    //nn.updateWeights()

nn.network.forEach((level)=>{
    level.forEach((neuron)=>{
        neuron.input_weights.forEach((w)=>{
            console.log(neuron.id,": to ", w.to.id, ", from ",w.from.id, w.weight)
        })
    })
})


console.log(nn.evaluate(dt))


//console.log(nn.weights)
//console.log(nn.network[1][0]._input_weights)

//console.log(nn.network[nn.visible_layer_level][0]._input_weights)



