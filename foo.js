'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const express = require('express');
const app = express();
app.use((req, res, next) => {    
    res.set("Cross-Origin-Embedder-Policy", "require-corp");    
    res.set("Cross-Origin-Opener-Policy", "same-origin");    
    next();
});
app.use(express.static('.'));
app.use(express.text({limit: '50mb'}));
/*app.post('*', (req, res) => {    
    const b = req.body;    
    const fname = path.basename(req.url);    
    //const time = (new Date()).toISOString();    
    //const fname = time.split(':').join('-');    
    //fs.writeFile(`post/${fname}.json`, b, () => void 0);    
    fs.writeFile(`post/${fname}`, b, () => void 0)    
    res.sendStatus(200);    
    res.end();
});*/

const credential = {    
    key: fs.readFileSync(path.resolve(__dirname, "key/key.pem")),    
    cert: fs.readFileSync(path.resolve(__dirname, "key/cert.pem")),
};
const server = https.createServer(credential, app);
//const server = http.createServer(app);
server.listen(8443, () => {    
    console.log("serever is runing at port 8443");
});