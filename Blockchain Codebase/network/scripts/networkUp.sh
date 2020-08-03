export PATH=$(pwd)/../bin:$PATH
export FABRIC_CFG_PATH=$(pwd)/configtx/

docker-compose -f ./docker/docker-compose-ca.yaml up -d

. organizations/fabric-ca/registerEnroll.sh

sleep 10

createManufacturer

createBuyer

createInspector

createOrderer

./organizations/ccp-generate.sh

configtxgen -profile ThreeOrgsOrdererGenesis -channelID system-channel -outputBlock ./system-genesis-block/genesis.block

docker-compose -f ./docker/docker-compose-couch.yaml -f ./docker/docker-compose-test-net.yaml up -d