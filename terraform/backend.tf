terraform {
  backend "azurerm" {
    resource_group_name   = "14185-irmai-1-jg5p49"
    storage_account_name  = "irmaitfstorage"
    container_name        = "terraform-state"
    key                   = "infra.tfstate"
  }
}
