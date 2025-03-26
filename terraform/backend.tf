terraform {
  backend "azurerm" {
    resource_group_name   = "14185-irmai-1-jg5p49"  # Your existing resource group
    storage_account_name  = "irmaitfstorage"       # Your storage account name
    container_name        = "terraform-state"      # Your container name
    key                   = "infra.tfstate"
  }
}
