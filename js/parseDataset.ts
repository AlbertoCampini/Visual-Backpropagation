import {Dataset} from "./Dataset";

export
function parseDataset(input): Dataset {
    let input_data = []
    let target_data = []
    input.DataSet.data.forEach((data)=>{
        input_data.push(data.input)
        target_data.push(data.target)
    })
    return new Dataset(input_data,target_data)
}