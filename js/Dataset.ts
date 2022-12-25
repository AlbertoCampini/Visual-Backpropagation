export class Dataset{
    input: any[][] = []
    target: any[][] = []


    constructor(input: any[], target: any[]) {
        this.input = input;
        this.target = target;
    }
}