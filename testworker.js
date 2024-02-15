for(let i=0; i<4; i++) {
    let worker = new Worker('worker.js');
    worker.onmessage = (e) => {
        console.log(e);
        if (e.data === 'start computing') {
            worker.postMessage(i);
        }
        
    }
}

// myWorker.terminate();
