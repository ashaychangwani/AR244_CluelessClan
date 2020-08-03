/*
 * Copyright IBM Corp. All Rights Reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const express = require('express');
const app = express();

app.use(express.static('public'))
app.set('view engine','pug');

var bodyParser = require('body-parser');
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies

const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');

var gateway;
var contract;

async function createGateway() {
    try {
        // load the network configuration
        const ccpPath = path.resolve(__dirname,'..','..', '..', 'network', 'organizations', 'peerOrganizations', 'manufacturer.example.com', 'connection-manufacturer.json');
        let ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new file system based wallet for managing identities.
        const walletPath = path.join(process.cwd(), 'wallet');
        const wallet = await Wallets.newFileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check to see if we've already enrolled the user.
        const identity = await wallet.get('appUser');
        if (!identity) {
            console.log('An identity for the user "appUser" does not exist in the wallet');
            console.log('Run the registerUser.js application before retrying');
            return;
        }

        // Create a new gateway for connecting to our peer node.
        gateway = new Gateway();
        await gateway.connect(ccp, { wallet, identity: 'appUser', discovery: { enabled: true, asLocalhost: true } });

        // Get the network (channel) our contract is deployed to.
        const network = await gateway.getNetwork('mychannel');

        // Get the contract from the network.
        contract = network.getContract('fabjute');
              
        //console.log('Asset Transfered');
      } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}



app.get('/', async function(req,res){

  await createGateway();

  result = await contract.evaluateTransaction("QueryAllAssets");
  result = result.toString();
  result = JSON.parse(result);
  console.log(result);


  jsonArray = result;
  res.render("dashboard.pug",{
    jsonArray:jsonArray
  });

  await gateway.disconnect();

});

app.listen(process.env.port || 3002);
console.log('Web Server is listening at port '+ (process.env.port || 3000));