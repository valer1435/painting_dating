
const express = require("express");
const app = express();

app.use(express.static(__dirname + '/frontend'));
app.get("/", function(request, response){
    response.sendFile(__dirname + "/frontend/main.html");
});

app.listen(3000);