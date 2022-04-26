const py_video = () => {
    eel.video_feed()();
}

const closeConnection = () => {
    const elem = document.getElementById("bg");
    eel.Close()();
    elem.src = "./Images/Nassim.jpeg"
}

const Temrinate = () => {
    eel.Close()();
}

const TrainModel = async () => {
    let info = document.getElementById("info");
    info.style.visibility = "visible";
    var n = await eel.train(50)();
    info.innerHTML = n;
}

eel.expose(updateImageSrc);
function updateImageSrc(val) {
    let elem = document.getElementById("bg");
    elem.src = "data:image/jpeg;base64," + val;
}

const TrainModelNeural = () => {
    eel.neuralNet()();
}

const TrainModelSuper = () => {
    eel.supervisedLearning()();
}

const TrainModelUnsper = () => {
    eel.UnsupervisedLearning()();
}

const TrainModelLinearOne = () => {
    eel.linearRegressionOne()();
}

const TrainModelMultiLinear = () => {
    eel.linearRegressionMulti()();
}

const KMEANS = () => {
    eel.kmeansMethod()();
}