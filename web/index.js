const input = document.getElementsByName("iterations");

const py_video = () => {
    eel.video_feed()();
}

const closeConnection = () => {
    let elem = document.getElementById("bg");
    eel.Close()();
    elem.src = "./Images/placeholder.png";
}

const MoveToTrain = () => {
    eel.Close()();
}

const TrainModel = () => {
    let info = document.getElementById("info");
    info.style.visibility = "visible";
    eel.train(input[0]?.value)();
}

eel.expose(updateImageSrc);
function updateImageSrc(val) {
    let elem = document.getElementById("bg");
    elem.src = "data:image/jpeg;base64," + val;
}