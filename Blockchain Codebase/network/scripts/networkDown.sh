docker-compose -f ./docker/docker-compose-ca.yaml -f ./docker/docker-compose-couch.yaml -f ./docker/docker-compose-test-net.yaml down --volumes --remove-orphans

rm -rf ./organizations/peerOrganizations/ ./organizations/ordererOrganizations/

rm -rf ./organizations/fabric-ca/buyer ./organizations/fabric-ca/*

rm -rf ./channel-artifacts/ ./system-genesis-block/ fabcar.tar.gz

pushd ../app/buyer/javascript/wallet/
    rm -rf *.id 
popd 

pushd ../app/manufacturer/javascript/wallet/
    rm -rf *.id  
popd 

pushd ../app/inspector/javascript/wallet/
    rm -rf *.id  
popd

cp -ar ./organizations/backup/. ./organizations/fabric-ca/