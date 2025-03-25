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
  subscription_id = "36448a90-905c-4f48-b1b3-deb171f7c247
"
  tenant_id       = "38b2a0a6-29a4-4744-ac80-7617428c3bbeD"
}

resource "azurerm_resource_group" "example" {
  name     = "backend-resources"
  location = "US East"
}
