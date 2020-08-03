const express = require('express');
const app = express();

app.use(express.static('public'))
app.set('view engine','pug');

var bodyParser = require('body-parser');
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies

jsonArray = 
[
  {
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute1",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  },{
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute2",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  },{
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute3",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  },{
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute4",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  },{
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute5",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  },{
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute6",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  },{
    "Name":"Jute",
    "Code":"16",
    "ID":"Jute7",
    "Owner":"Manufacturer",
    "Manufacturer":"Manufacturer",
    "Status":"Not Inspected"
  }
]


app.get('/', (req,res) => {
  res.render("dashboard.pug",{
    jsonArray:jsonArray
  });
});

app.post('/createAsset',(req,res) => {

  var newObj = {
    "Name": "Jute",
    "Code": "16",
    "ID": req.body.assetid,
    "Owner": "Manufacturer",
    "Manufacturer": "Manufacturer",
    "StatusID": "Not Inspected",
  }
  jsonArray.push(newObj);
  res.redirect('/');
});

app.post('/transferAsset',(req,res) => {
  res.redirect('/');
});

app.post('/transferResponsibility',(req,res) => {
  res.redirect('/');
});

app.post('/inspectAsset',(req,res) => {
  res.redirect('/');
});

app.get('/getAssetHistory/:id',(req,res) => {
  res.send("Hello");
});

app.listen(process.env.port || 3000);
console.log('Web Server is listening at port '+ (process.env.port || 3000));