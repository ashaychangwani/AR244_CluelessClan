#!/bin/bash
export PATH=$(pwd)/../bin:$PATH
export FABRIC_CFG_PATH=$(pwd)/configtxgen

CHANNEL_NAME="$1"
DELAY="$2"
MAX_RETRY="$3"
VERBOSE="$4"
: ${CHANNEL_NAME:="mychannel"}
: ${DELAY:="3"}
: ${MAX_RETRY:="5"}
: ${VERBOSE:="true"}

if [ ! -d "channel-artifacts" ]; then
	mkdir channel-artifacts
fi

createChannelTx() {
	configtxgen -profile ThreeOrgsChannel -outputCreateChannelTx ./channel-artifacts/${CHANNEL_NAME}.tx -channelID $CHANNEL_NAME
}

createAncorPeerTx() {

	for orgmsp in manufacturerMSP buyerMSP inspectorMSP; do

	echo "#######    Generating anchor peer update transaction for ${orgmsp}  ##########"
	set -x
	configtxgen -profile ThreeOrgsChannel -outputAnchorPeersUpdate ./channel-artifacts/${orgmsp}anchors.tx -channelID $CHANNEL_NAME -asOrg ${orgmsp}
	res=$?
	done
}

createChannel() {

	peer channel create -o localhost:7050 -c $CHANNEL_NAME --ordererTLSHostnameOverride orderer.example.com -f ./channel-artifacts/${CHANNEL_NAME}.tx --outputBlock ./channel-artifacts/${CHANNEL_NAME}.block --tls --cafile $ORDERER_CA >&log.txt
}

joinChannel(){
	sleep $DELAY
	peer channel join -b ./channel-artifacts/$CHANNEL_NAME.block
}

updateAnchorPeers(){
	FABRIC_CFG_PATH=$PWD/../config/
	peer channel update -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com -c $CHANNEL_NAME -f ./channel-artifacts/${CORE_PEER_LOCALMSPID}anchors.tx --tls --cafile $ORDERER_CA
}

ordererFunc(){
	FABRIC_CFG_PATH=${PWD}/configtx
	createChannelTx
	createAncorPeerTx
}

manufacturerFunc(){
	FABRIC_CFG_PATH=$PWD/../config/
	createChannel
	joinChannel
}

buyerFunc(){
	FABRIC_CFG_PATH=$PWD/../config/
	joinChannel
}

inspectorFunc(){
	FABRIC_CFG_PATH=$PWD/../config/
	joinChannel
}
