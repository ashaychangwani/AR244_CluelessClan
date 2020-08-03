CC_SRC_PATH="../fabjute/"
CC_NAME="fabjute"
CC_RUNTIME_LANGUAGE="golang"
CC_VERSION="1"
INIT_REQUIRED="--init-required"
CC_INIT_FCN="initLedger"
INIT_REQUIRED=""
CC_INIT_FCN=""

CC_COLL_CONFIG="--collections-config ../fabjute/collections_config.json"
CC_END_POLICY=""
CC_SEQUENCE="1"
DELAY="3"
CHANNEL_NAME="mychannel"

CORE_PEER_ADDRESS_MANUFACTURER=localhost:7051
CORE_PEER_TLS_ROOTCERT_FILE_MANUFACTURER=$(pwd)/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/ca.crt

CORE_PEER_ADDRESS_BUYER=localhost:9051
CORE_PEER_TLS_ROOTCERT_FILE_BUYER=$(pwd)/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/ca.crt

CORE_PEER_ADDRESS_INSPECTOR=localhost:11051
CORE_PEER_TLS_ROOTCERT_FILE_INSPECTOR=$(pwd)/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/ca.crt

installDependencies(){
    pushd $CC_SRC_PATH
        GO111MODULE=on go mod vendor
    popd
}

packageChaincode(){ 
    peer lifecycle chaincode package ${CC_NAME}.tar.gz --path ${CC_SRC_PATH} --lang ${CC_RUNTIME_LANGUAGE} --label ${CC_NAME}_${CC_VERSION} 
}

installChaincode(){
    peer lifecycle chaincode install ${CC_NAME}.tar.gz
}

queryInstalled(){
    peer lifecycle chaincode queryinstalled >&log.txt
    cat log.txt
	PACKAGE_ID=$(sed -n "/${CC_NAME}_${CC_VERSION}/{s/^Package ID: //; s/, Label:.*$//; p;}" log.txt)
}

approveForMyOrg(){
    peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile $ORDERER_CA --channelID $CHANNEL_NAME --name ${CC_NAME} --version ${CC_VERSION} --package-id ${PACKAGE_ID} --sequence ${CC_SEQUENCE} ${INIT_REQUIRED} ${CC_END_POLICY} ${CC_COLL_CONFIG}
}

checkCommitReadiness() {
    sleep $DELAY
    peer lifecycle chaincode checkcommitreadiness --channelID $CHANNEL_NAME --name ${CC_NAME} --version ${CC_VERSION} --sequence ${CC_SEQUENCE} ${INIT_REQUIRED} ${CC_END_POLICY} ${CC_COLL_CONFIG} --output json 
}

commitChaincodeDefinition(){
    sleep $DELAY
    peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile $ORDERER_CA --channelID $CHANNEL_NAME --name ${CC_NAME} --peerAddresses ${CORE_PEER_ADDRESS_MANUFACTURER} --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE_MANUFACTURER} --peerAddresses ${CORE_PEER_ADDRESS_BUYER} --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE_BUYER} --peerAddresses ${CORE_PEER_ADDRESS_INSPECTOR} --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE_INSPECTOR} --version ${CC_VERSION} --sequence ${CC_SEQUENCE} ${INIT_REQUIRED} ${CC_END_POLICY} ${CC_COLL_CONFIG} 
}

queryCommitted() {
    sleep $DELAY
    peer lifecycle chaincode querycommitted --channelID $CHANNEL_NAME --name ${CC_NAME}
}

chaincodeInvokeInit(){
    fcn_call='{"Args":["initLedger"]}'
    peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile $ORDERER_CA -C $CHANNEL_NAME -n ${CC_NAME} --peerAddresses ${CORE_PEER_ADDRESS_MANUFACTURER} --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE_MANUFACTURER} --peerAddresses ${CORE_PEER_ADDRESS_BUYER} --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE_BUYER} --peerAddresses ${CORE_PEER_ADDRESS_INSPECTOR} --tlsRootCertFiles ${CORE_PEER_TLS_ROOTCERT_FILE_INSPECTOR} -c ${fcn_call} 
}

chaincodeQuery() {
    peer chaincode query -C $CHANNEL_NAME -n ${CC_NAME} -c '{"Args":["QueryAllAssets"]}'
}

manufacturerFunc(){
    installDependencies
    packageChaincode
    installChaincode
    queryInstalled
    approveForMyOrg
}

buyerFunc(){
    installChaincode
    queryInstalled
    approveForMyOrg
}

inspectorFunc(){
    installChaincode
    queryInstalled
    approveForMyOrg
}