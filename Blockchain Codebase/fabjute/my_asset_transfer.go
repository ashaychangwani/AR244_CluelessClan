package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strconv"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type Asset struct {
	AssetNumber  string `json:"assetnumber"`
	AssetId      string `json:"assetid"`
	Manufacturer string `json:"manufacturer"`
	Owner        string `json:"owner"`
	Status       string `json:"status"`
	Handler      string `json:"handler"`
	Buyer        string `json:"buyer"`

	Location     string `json:"location"`
	Inspector    string `json:"inspector"`
	Description  string `json:"description"`
	GAW          string `json:"gaw"`
	Weft         string `json:"weft"`
	Cut          string `json:"cut"`
	MajorDefects string `json:"majordefects"`
	MinorDefects string `json:"minordefects`
}

type SmartContract struct {
	contractapi.Contract
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {

	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get verified OrgID: %v", err)
	}

	for i := 0; i < 10; i++ {
		var assetId = createAssetId(strconv.Itoa(i), clientMSPID)

		asset := Asset{
			AssetNumber:  strconv.Itoa(i),
			AssetId:      assetId,
			Manufacturer: clientMSPID,
			Owner:        clientMSPID,
			Status:       "Sale",
			Handler:      "None",
			Buyer:        "None",
			Location:     "Delhi",
			Inspector:    "Inspector",
			Description:  "Ipsem Lorem",
			GAW:          "0.7",
			Weft:         "Two or More",
			Cut:          "Many",
			MajorDefects: "5",
			MinorDefects: "9",
		}

		assetAsBytes, _ := json.Marshal(asset)
		err := ctx.GetStub().PutState(asset.AssetId, assetAsBytes)

		if err != nil {
			return fmt.Errorf("Failed to put to world state. %s,asset number %s", err.Error(), i)
		}
	}

	return nil
}

func (s *SmartContract) CreateAsset(ctx contractapi.TransactionContextInterface, assetNumber string, description string, gaw string, weft string,
	cut string, majorDefects string, minorDefects string) error {
	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get verified OrgID: %v", err)
	}

	var assetID = createAssetId(assetNumber, clientMSPID)

	// Creating asset Asset
	asset := Asset{
		AssetNumber:  assetNumber,
		AssetId:      assetID,
		Manufacturer: clientMSPID,
		Owner:        clientMSPID,
		Status:       "Sale",
		Handler:      "None",
		Buyer:        "None",
		Location:     "Delhi",
		Inspector:    "Inspector",
		Description:  description,
		GAW:          gaw,
		Weft:         weft,
		Cut:          cut,
		MajorDefects: majorDefects,
		MinorDefects: minorDefects,
	}

	//converting to bytes
	assetBytes, err := json.Marshal(asset)
	if err != nil {
		return fmt.Errorf("failed to create asset JSON: %v", err)
	}

	err = ctx.GetStub().PutState(asset.AssetId, assetBytes)
	if err != nil {
		return fmt.Errorf("failed to put asset in public data: %v", err)
	}

	return nil
}

func (s *SmartContract) TakeAStop(ctx contractapi.TransactionContextInterface, assetID string, location string, handler string, status string) error {

	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()

	asset, err := s.ReadAsset(ctx, assetID)
	if err != nil {
		return fmt.Errorf("Asset not found")
	}

	if asset.Owner != clientMSPID {
		return fmt.Errorf("Owner mismatch")
	}

	handlerprevious := ""
	if status != "ok" {
		handlerprevious = asset.Handler
		asset.Status = "Damaged"
	}

	asset.Handler = handler
	asset.Location = location

	assetBytes, err := json.Marshal(asset)
	if err != nil {
		return fmt.Errorf("failed to create asset JSON: %v", err)
	}

	err = ctx.GetStub().PutState(asset.AssetId, assetBytes)
	if err != nil {
		return fmt.Errorf("failed to put asset in public data: %v", err)
	}

	if asset.Status == "Damaged" {
		return fmt.Errorf("Asset has been damaged by party %s Call back asset", handlerprevious)
	}

	return nil
}

func (s *SmartContract) AssetReceived(ctx contractapi.TransactionContextInterface, assetID string) error {

	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("Get Buyer ID")
	}

	asset, err := s.ReadAsset(ctx, assetID)
	if err != nil {
		return fmt.Errorf("Asset not found")
	}

	if asset.Buyer != clientMSPID {
		return fmt.Errorf("Wrong buyer")
	}

	asset.Owner = asset.Buyer
	asset.Buyer = "None"

	assetBytes, err := json.Marshal(asset)
	if err != nil {
		return fmt.Errorf("failed to create asset JSON: %v", err)
	}

	err = ctx.GetStub().PutState(asset.AssetId, assetBytes)
	if err != nil {
		return fmt.Errorf("failed to put asset in public data: %v", err)
	}

	return nil
}

func (s *SmartContract) BuyAsset(ctx contractapi.TransactionContextInterface, assetID string) error {

	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("Cant get Buyer ID")
	}

	asset, err := s.ReadAsset(ctx, assetID)
	if err != nil {
		return fmt.Errorf("Asset not found")
	}

	if asset.Status != "Sale" {
		return fmt.Errorf("Asset not for sale")
	}

	asset.Status = "ok"
	asset.Buyer = clientMSPID

	assetBytes, err := json.Marshal(asset)
	if err != nil {
		return fmt.Errorf("failed to create asset JSON: %v", err)
	}

	err = ctx.GetStub().PutState(asset.AssetId, assetBytes)
	if err != nil {
		return fmt.Errorf("failed to put asset in public data: %v", err)
	}

	return nil
}

func createAssetId(assetNumber string, clientMSPID string) string {
	assetId := "Asset-" + clientMSPID + assetNumber
	return (assetId)
}

func main() {

	chaincode, err := contractapi.NewChaincode(new(SmartContract))

	if err != nil {
		log.Panicf("error creating chaincode: %v", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		log.Panicf("error starting chaincode: %v", err)
	}
}
