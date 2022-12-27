export
const ACTIVATION_FUNCTION: string = "sigmoid"

export
function getRandomArbitrary(min = -1, max = 1) {
    return Math.random() * (max - min) + min;
}

export
function fixNumber(number,decimal= 2): number{
    return Math.round((number + Number.EPSILON) * Math.pow(10,decimal)) / Math.pow(10,decimal)
}