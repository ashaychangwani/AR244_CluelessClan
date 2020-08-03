

function createManufacturer {

  echo
	echo "Enroll the CA admin"
  echo
	mkdir -p organizations/peerOrganizations/manufacturer.example.com/

	export FABRIC_CA_CLIENT_HOME=${PWD}/organizations/peerOrganizations/manufacturer.example.com/
  #  rm -rf $FABRIC_CA_CLIENT_HOME/fabric-ca-client-config.yaml
  #  rm -rf $FABRIC_CA_CLIENT_HOME/msp

  set -x
  fabric-ca-client enroll -u https://admin:adminpw@localhost:7054 --caname ca-manufacturer --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

  echo 'NodeOUs:
  Enable: true
  ClientOUIdentifier:
    Certificate: cacerts/localhost-7054-ca-manufacturer.pem
    OrganizationalUnitIdentifier: client
  PeerOUIdentifier:
    Certificate: cacerts/localhost-7054-ca-manufacturer.pem
    OrganizationalUnitIdentifier: peer
  AdminOUIdentifier:
    Certificate: cacerts/localhost-7054-ca-manufacturer.pem
    OrganizationalUnitIdentifier: admin
  OrdererOUIdentifier:
    Certificate: cacerts/localhost-7054-ca-manufacturer.pem
    OrganizationalUnitIdentifier: orderer' > ${PWD}/organizations/peerOrganizations/manufacturer.example.com/msp/config.yaml

  echo
	echo "Register peer0"
  echo
  set -x
	fabric-ca-client register --caname ca-manufacturer --id.name peer0 --id.secret peer0pw --id.type peer --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

  echo
  echo "Register user"
  echo
  set -x
  fabric-ca-client register --caname ca-manufacturer --id.name user1 --id.secret user1pw --id.type client --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

  echo
  echo "Register the org admin"
  echo
  set -x
  fabric-ca-client register --caname ca-manufacturer --id.name manufactureradmin --id.secret manufactureradminpw --id.type admin --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

	mkdir -p organizations/peerOrganizations/manufacturer.example.com/peers
  mkdir -p organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com

  echo
  echo "## Generate the peer0 msp"
  echo
  set -x
	fabric-ca-client enroll -u https://peer0:peer0pw@localhost:7054 --caname ca-manufacturer -M ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/msp --csr.hosts peer0.manufacturer.example.com --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/msp/config.yaml

  echo
  echo "## Generate the peer0-tls certificates"
  echo
  set -x
  fabric-ca-client enroll -u https://peer0:peer0pw@localhost:7054 --caname ca-manufacturer -M ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls --enrollment.profile tls --csr.hosts peer0.manufacturer.example.com --csr.hosts localhost --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x


  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/ca.crt
  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/signcerts/* ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/server.crt
  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/keystore/* ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/server.key

  mkdir -p ${PWD}/organizations/peerOrganizations/manufacturer.example.com/msp/tlscacerts
  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/manufacturer.example.com/msp/tlscacerts/ca.crt

  mkdir -p ${PWD}/organizations/peerOrganizations/manufacturer.example.com/tlsca
  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/manufacturer.example.com/tlsca/tlsca.manufacturer.example.com-cert.pem

  mkdir -p ${PWD}/organizations/peerOrganizations/manufacturer.example.com/ca
  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/peers/peer0.manufacturer.example.com/msp/cacerts/* ${PWD}/organizations/peerOrganizations/manufacturer.example.com/ca/ca.manufacturer.example.com-cert.pem

  mkdir -p organizations/peerOrganizations/manufacturer.example.com/users
  mkdir -p organizations/peerOrganizations/manufacturer.example.com/users/User1@manufacturer.example.com

  echo
  echo "## Generate the user msp"
  echo
  set -x
	fabric-ca-client enroll -u https://user1:user1pw@localhost:7054 --caname ca-manufacturer -M ${PWD}/organizations/peerOrganizations/manufacturer.example.com/users/User1@manufacturer.example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/manufacturer.example.com/users/User1@manufacturer.example.com/msp/config.yaml

  mkdir -p organizations/peerOrganizations/manufacturer.example.com/users/Admin@manufacturer.example.com

  echo
  echo "## Generate the org admin msp"
  echo
  set -x
	fabric-ca-client enroll -u https://manufactureradmin:manufactureradminpw@localhost:7054 --caname ca-manufacturer -M ${PWD}/organizations/peerOrganizations/manufacturer.example.com/users/Admin@manufacturer.example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/manufacturer/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/manufacturer.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/manufacturer.example.com/users/Admin@manufacturer.example.com/msp/config.yaml

}

function createBuyer {

  echo
	echo "Enroll the CA admin"
  echo
	mkdir -p organizations/peerOrganizations/buyer.example.com/

	export FABRIC_CA_CLIENT_HOME=${PWD}/organizations/peerOrganizations/buyer.example.com/
  #  rm -rf $FABRIC_CA_CLIENT_HOME/fabric-ca-client-config.yaml
  #  rm -rf $FABRIC_CA_CLIENT_HOME/msp

  set -x
  fabric-ca-client enroll -u https://admin:adminpw@localhost:8054 --caname ca-buyer --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

  echo 'NodeOUs:
  Enable: true
  ClientOUIdentifier:
    Certificate: cacerts/localhost-8054-ca-buyer.pem
    OrganizationalUnitIdentifier: client
  PeerOUIdentifier:
    Certificate: cacerts/localhost-8054-ca-buyer.pem
    OrganizationalUnitIdentifier: peer
  AdminOUIdentifier:
    Certificate: cacerts/localhost-8054-ca-buyer.pem
    OrganizationalUnitIdentifier: admin
  OrdererOUIdentifier:
    Certificate: cacerts/localhost-8054-ca-buyer.pem
    OrganizationalUnitIdentifier: orderer' > ${PWD}/organizations/peerOrganizations/buyer.example.com/msp/config.yaml

  echo
	echo "Register peer0"
  echo
  set -x
	fabric-ca-client register --caname ca-buyer --id.name peer0 --id.secret peer0pw --id.type peer --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

  echo
  echo "Register user"
  echo
  set -x
  fabric-ca-client register --caname ca-buyer --id.name user1 --id.secret user1pw --id.type client --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

  echo
  echo "Register the org admin"
  echo
  set -x
  fabric-ca-client register --caname ca-buyer --id.name buyeradmin --id.secret buyeradminpw --id.type admin --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

	mkdir -p organizations/peerOrganizations/buyer.example.com/peers
  mkdir -p organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com

  echo
  echo "## Generate the peer0 msp"
  echo
  set -x
	fabric-ca-client enroll -u https://peer0:peer0pw@localhost:8054 --caname ca-buyer -M ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/msp --csr.hosts peer0.buyer.example.com --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/msp/config.yaml

  echo
  echo "## Generate the peer0-tls certificates"
  echo
  set -x
  fabric-ca-client enroll -u https://peer0:peer0pw@localhost:8054 --caname ca-buyer -M ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls --enrollment.profile tls --csr.hosts peer0.buyer.example.com --csr.hosts localhost --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x


  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/ca.crt
  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/signcerts/* ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/server.crt
  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/keystore/* ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/server.key

  mkdir -p ${PWD}/organizations/peerOrganizations/buyer.example.com/msp/tlscacerts
  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/buyer.example.com/msp/tlscacerts/ca.crt

  mkdir -p ${PWD}/organizations/peerOrganizations/buyer.example.com/tlsca
  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/buyer.example.com/tlsca/tlsca.buyer.example.com-cert.pem

  mkdir -p ${PWD}/organizations/peerOrganizations/buyer.example.com/ca
  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/peers/peer0.buyer.example.com/msp/cacerts/* ${PWD}/organizations/peerOrganizations/buyer.example.com/ca/ca.buyer.example.com-cert.pem

  mkdir -p organizations/peerOrganizations/buyer.example.com/users
  mkdir -p organizations/peerOrganizations/buyer.example.com/users/User1@buyer.example.com

  echo
  echo "## Generate the user msp"
  echo
  set -x
	fabric-ca-client enroll -u https://user1:user1pw@localhost:8054 --caname ca-buyer -M ${PWD}/organizations/peerOrganizations/buyer.example.com/users/User1@buyer.example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/buyer.example.com/users/User1@buyer.example.com/msp/config.yaml

  mkdir -p organizations/peerOrganizations/buyer.example.com/users/Admin@buyer.example.com

  echo
  echo "## Generate the org admin msp"
  echo
  set -x
	fabric-ca-client enroll -u https://buyeradmin:buyeradminpw@localhost:8054 --caname ca-buyer -M ${PWD}/organizations/peerOrganizations/buyer.example.com/users/Admin@buyer.example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/buyer/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/buyer.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/buyer.example.com/users/Admin@buyer.example.com/msp/config.yaml

}

function createInspector {

  echo
	echo "Enroll the CA admin"
  echo
	mkdir -p organizations/peerOrganizations/inspector.example.com/

	export FABRIC_CA_CLIENT_HOME=${PWD}/organizations/peerOrganizations/inspector.example.com/
  #  rm -rf $FABRIC_CA_CLIENT_HOME/fabric-ca-client-config.yaml
  #  rm -rf $FABRIC_CA_CLIENT_HOME/msp

  set -x
  fabric-ca-client enroll -u https://admin:adminpw@localhost:11054 --caname ca-inspector --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

  echo 'NodeOUs:
  Enable: true
  ClientOUIdentifier:
    Certificate: cacerts/localhost-11054-ca-inspector.pem
    OrganizationalUnitIdentifier: client
  PeerOUIdentifier:
    Certificate: cacerts/localhost-11054-ca-inspector.pem
    OrganizationalUnitIdentifier: peer
  AdminOUIdentifier:
    Certificate: cacerts/localhost-11054-ca-inspector.pem
    OrganizationalUnitIdentifier: admin
  OrdererOUIdentifier:
    Certificate: cacerts/localhost-11054-ca-inspector.pem
    OrganizationalUnitIdentifier: orderer' > ${PWD}/organizations/peerOrganizations/inspector.example.com/msp/config.yaml

  echo
	echo "Register peer0"
  echo
  set -x
	fabric-ca-client register --caname ca-inspector --id.name peer0 --id.secret peer0pw --id.type peer --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

  echo
  echo "Register user"
  echo
  set -x
  fabric-ca-client register --caname ca-inspector --id.name user1 --id.secret user1pw --id.type client --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

  echo
  echo "Register the org admin"
  echo
  set -x
  fabric-ca-client register --caname ca-inspector --id.name inspectoradmin --id.secret inspectoradminpw --id.type admin --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

	mkdir -p organizations/peerOrganizations/inspector.example.com/peers
  mkdir -p organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com

  echo
  echo "## Generate the peer0 msp"
  echo
  set -x
	fabric-ca-client enroll -u https://peer0:peer0pw@localhost:11054 --caname ca-inspector -M ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/msp --csr.hosts peer0.inspector.example.com --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/msp/config.yaml

  echo
  echo "## Generate the peer0-tls certificates"
  echo
  set -x
  fabric-ca-client enroll -u https://peer0:peer0pw@localhost:11054 --caname ca-inspector -M ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls --enrollment.profile tls --csr.hosts peer0.inspector.example.com --csr.hosts localhost --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x


  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/ca.crt
  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/signcerts/* ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/server.crt
  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/keystore/* ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/server.key

  mkdir -p ${PWD}/organizations/peerOrganizations/inspector.example.com/msp/tlscacerts
  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/inspector.example.com/msp/tlscacerts/ca.crt

  mkdir -p ${PWD}/organizations/peerOrganizations/inspector.example.com/tlsca
  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/tls/tlscacerts/* ${PWD}/organizations/peerOrganizations/inspector.example.com/tlsca/tlsca.inspector.example.com-cert.pem

  mkdir -p ${PWD}/organizations/peerOrganizations/inspector.example.com/ca
  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/peers/peer0.inspector.example.com/msp/cacerts/* ${PWD}/organizations/peerOrganizations/inspector.example.com/ca/ca.inspector.example.com-cert.pem

  mkdir -p organizations/peerOrganizations/inspector.example.com/users
  mkdir -p organizations/peerOrganizations/inspector.example.com/users/User1@inspector.example.com

  echo
  echo "## Generate the user msp"
  echo
  set -x
	fabric-ca-client enroll -u https://user1:user1pw@localhost:11054 --caname ca-inspector -M ${PWD}/organizations/peerOrganizations/inspector.example.com/users/User1@inspector.example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/inspector.example.com/users/User1@inspector.example.com/msp/config.yaml

  mkdir -p organizations/peerOrganizations/inspector.example.com/users/Admin@inspector.example.com

  echo
  echo "## Generate the org admin msp"
  echo
  set -x
	fabric-ca-client enroll -u https://inspectoradmin:inspectoradminpw@localhost:11054 --caname ca-inspector -M ${PWD}/organizations/peerOrganizations/inspector.example.com/users/Admin@inspector.example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/inspector/tls-cert.pem
  set +x

  cp ${PWD}/organizations/peerOrganizations/inspector.example.com/msp/config.yaml ${PWD}/organizations/peerOrganizations/inspector.example.com/users/Admin@inspector.example.com/msp/config.yaml

}

function createOrderer {

  echo
	echo "Enroll the CA admin"
  echo
	mkdir -p organizations/ordererOrganizations/example.com

	export FABRIC_CA_CLIENT_HOME=${PWD}/organizations/ordererOrganizations/example.com
  #  rm -rf $FABRIC_CA_CLIENT_HOME/fabric-ca-client-config.yaml
  #  rm -rf $FABRIC_CA_CLIENT_HOME/msp

  set -x
  fabric-ca-client enroll -u https://admin:adminpw@localhost:9054 --caname ca-orderer --tls.certfiles ${PWD}/organizations/fabric-ca/ordererOrg/tls-cert.pem
  set +x

  echo 'NodeOUs:
  Enable: true
  ClientOUIdentifier:
    Certificate: cacerts/localhost-9054-ca-orderer.pem
    OrganizationalUnitIdentifier: client
  PeerOUIdentifier:
    Certificate: cacerts/localhost-9054-ca-orderer.pem
    OrganizationalUnitIdentifier: peer
  AdminOUIdentifier:
    Certificate: cacerts/localhost-9054-ca-orderer.pem
    OrganizationalUnitIdentifier: admin
  OrdererOUIdentifier:
    Certificate: cacerts/localhost-9054-ca-orderer.pem
    OrganizationalUnitIdentifier: orderer' > ${PWD}/organizations/ordererOrganizations/example.com/msp/config.yaml


  echo
	echo "Register orderer"
  echo
  set -x
	fabric-ca-client register --caname ca-orderer --id.name orderer --id.secret ordererpw --id.type orderer --tls.certfiles ${PWD}/organizations/fabric-ca/ordererOrg/tls-cert.pem
    set +x

  echo
  echo "Register the orderer admin"
  echo
  set -x
  fabric-ca-client register --caname ca-orderer --id.name ordererAdmin --id.secret ordererAdminpw --id.type admin --tls.certfiles ${PWD}/organizations/fabric-ca/ordererOrg/tls-cert.pem
  set +x

	mkdir -p organizations/ordererOrganizations/example.com/orderers
  mkdir -p organizations/ordererOrganizations/example.com/orderers/example.com

  mkdir -p organizations/ordererOrganizations/example.com/orderers/orderer.example.com

  echo
  echo "## Generate the orderer msp"
  echo
  set -x
	fabric-ca-client enroll -u https://orderer:ordererpw@localhost:9054 --caname ca-orderer -M ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp --csr.hosts orderer.example.com --csr.hosts localhost --tls.certfiles ${PWD}/organizations/fabric-ca/ordererOrg/tls-cert.pem
  set +x

  cp ${PWD}/organizations/ordererOrganizations/example.com/msp/config.yaml ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/config.yaml

  echo
  echo "## Generate the orderer-tls certificates"
  echo
  set -x
  fabric-ca-client enroll -u https://orderer:ordererpw@localhost:9054 --caname ca-orderer -M ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls --enrollment.profile tls --csr.hosts orderer.example.com --csr.hosts localhost --tls.certfiles ${PWD}/organizations/fabric-ca/ordererOrg/tls-cert.pem
  set +x

  cp ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/tlscacerts/* ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt
  cp ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/signcerts/* ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/server.crt
  cp ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/keystore/* ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/server.key

  mkdir -p ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts
  cp ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/tlscacerts/* ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

  mkdir -p ${PWD}/organizations/ordererOrganizations/example.com/msp/tlscacerts
  cp ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/tlscacerts/* ${PWD}/organizations/ordererOrganizations/example.com/msp/tlscacerts/tlsca.example.com-cert.pem

  mkdir -p organizations/ordererOrganizations/example.com/users
  mkdir -p organizations/ordererOrganizations/example.com/users/Admin@example.com

  echo
  echo "## Generate the admin msp"
  echo
  set -x
	fabric-ca-client enroll -u https://ordererAdmin:ordererAdminpw@localhost:9054 --caname ca-orderer -M ${PWD}/organizations/ordererOrganizations/example.com/users/Admin@example.com/msp --tls.certfiles ${PWD}/organizations/fabric-ca/ordererOrg/tls-cert.pem
  set +x

  cp ${PWD}/organizations/ordererOrganizations/example.com/msp/config.yaml ${PWD}/organizations/ordererOrganizations/example.com/users/Admin@example.com/msp/config.yaml


}
