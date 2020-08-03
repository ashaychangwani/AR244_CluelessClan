ordererFunc(){
    export CORE_PEER_LOCALMSPID="OrdererMSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem
    export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/ordererOrganizations/example.com/users/Admin@example.com/msp
}

manufacturerFunc(){
    export ORDERER_CA=$(pwd)/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem 
    export PATH=$(pwd)/../bin:$PATH 
    export FABRIC_CFG_PATH=$(pwd)/../config/ 
    export CORE_PEER_TLS_ENABLED=true 
    export CORE_PEER_LOCALMSPID="manufacturerMSP" 
    export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/manufacturer.example.com/users/Admin@manufacturer.example.com/msp 
    export CORE_PEER_ADDRESS=localhost:7051
}

buyerFunc(){
    export ORDERER_CA=$(pwd)/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem 
    export PATH=$(pwd)/../bin:$PATH 
    export FABRIC_CFG_PATH=$(pwd)/../config/ 
    export CORE_PEER_TLS_ENABLED=true 
    export CORE_PEER_LOCALMSPID="buyerMSP" 
    export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/buyer.example.com/users/Admin@buyer.example.com/msp 
    export CORE_PEER_ADDRESS=localhost:9051
}

inspectorFunc(){
    export ORDERER_CA=$(pwd)/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem 
    export PATH=$(pwd)/../bin:$PATH 
    export FABRIC_CFG_PATH=$(pwd)/../config/ 
    export CORE_PEER_TLS_ENABLED=true 
    export CORE_PEER_LOCALMSPID="inspectorMSP" 
    export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/inspector.example.com/users/Admin@inspector.example.com/msp 
    export CORE_PEER_ADDRESS=localhost:11051
}