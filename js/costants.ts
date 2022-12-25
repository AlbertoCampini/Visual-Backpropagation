export
const ACTIVATION_FUNCTION: string = "sigmoid"

export
function getRandomArbitrary(min = -0.5, max = 0.5) {
    return Math.random() * (max - min) + min;
}