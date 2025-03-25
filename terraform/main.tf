terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "backend-rg"
    storage_account_name = "irmaitfbackendstorage"
    container_name       = "terraform-state"
    key                  = "terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
  
}

resource "azurerm_resource_group" "example" {
  name     = "backend-resources"
  location = "US East"
}
