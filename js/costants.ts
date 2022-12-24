export
const ACTIVATION_FUNCTION: string = "sigmoid"

export
function getRandomArbitrary(min = -1, max = 1) {
    return Math.random() * (max - min) + min;
}