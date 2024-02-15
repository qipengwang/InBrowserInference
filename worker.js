let myid;

onmessage = (e) => {
    myid = e.data;
    let a = new Int32Array(100), b = new Int32Array(100);
    for (let i = 0; i < 100; i++) {
        a[i] = new Int32Array(100);
        b[i] = new Int32Array(100);
        for (let j = 0; j < 100; j++) {
            a[i][j] = 1;
            b[i][j] = 1;
        }
    }

    for (let iter = 1; iter > 0; iter++) {
        let c = new Int32Array(100);
        for (let i = 0; i < 100; i++) {
            c[i] = new Int32Array(100);
            for (let j = 0; j < 100; j++) {
                c[i][j] = iter;
            }
        }
        for (let i = 0; i < 100; i++) {
            for (let j = 0; j < 100; j++) {
                for (let k = 0; k < 100; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        postMessage(myid + ": finish iter " + iter);
    }
}

postMessage("start computing");



