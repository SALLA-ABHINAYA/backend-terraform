provider "azurerm" {
  features {}
  subscription_id = "36448a90-905c-4f48-b1b3-deb171f7c247" # Ensure this subscription ID is correct
}

# Use an existing Resource Group
data "azurerm_resource_group" "irmai_rg" {
  name = "14185-irmai-1-jg5p49" # Ensure this name matches the existing resource group
}

# Virtual Network
resource "azurerm_virtual_network" "irmai_vnet" {
  name                = "irmai-1-jg5p49-vn"
  address_space       = ["10.0.0.0/16"]
  location            = data.azurerm_resource_group.irmai_rg.location
  resource_group_name = data.azurerm_resource_group.irmai_rg.name
}

# Subnet
resource "azurerm_subnet" "irmai_subnet" {
  name                 = "irmai-1-jg5p49-sn"
  resource_group_name  = data.azurerm_resource_group.irmai_rg.name
  virtual_network_name = azurerm_virtual_network.irmai_vnet.name
  address_prefixes     = ["10.0.0.0/24"]
}

# Network Security Group (NSG) - Best practice default deny inbound
resource "azurerm_network_security_group" "irmai_nsg" {
  name                = "irmai-default-nsg"
  location            = data.azurerm_resource_group.irmai_rg.location
  resource_group_name = data.azurerm_resource_group.irmai_rg.name

  security_rule {
    name                       = "Allow-HTTPS-Out"
    priority                   = 100
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "Internet"
  }

  security_rule {
    name                       = "Deny-All-Inbound"
    priority                   = 4096
    direction                  = "Inbound"
    access                     = "Deny"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Associate NSG with Subnet
resource "azurerm_subnet_network_security_group_association" "irmai_subnet_nsg_assoc" {
  subnet_id                 = azurerm_subnet.irmai_subnet.id
  network_security_group_id = azurerm_network_security_group.irmai_nsg.id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "irmai_aks" {
  name                = "irmai-1-jg5p49"
  location            = data.azurerm_resource_group.irmai_rg.location
  resource_group_name = data.azurerm_resource_group.irmai_rg.name
  dns_prefix          = "irmai"

  default_node_pool {
    name       = "system"
    node_count = 1
    vm_size    = "Standard_B4as_v2"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin     = "azure"
    load_balancer_sku  = "standard"
    dns_service_ip     = "10.0.0.10"
    service_cidr       = "10.0.0.0/16"
  }
}

# Additional Node Pools
resource "azurerm_kubernetes_cluster_node_pool" "appvr" {
  name                  = "c4465appvr"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.irmai_aks.id
  vm_size               = "Standard_B4as_v2"
  node_count            = 3
  os_type               = "Linux"
  mode                  = "User"
}

resource "azurerm_kubernetes_cluster_node_pool" "azuk7" {
  name                  = "c4465azuk7"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.irmai_aks.id
  vm_size               = "Standard_B4as_v2"
  node_count            = 1
  os_type               = "Linux"
  mode                  = "User"
}

resource "azurerm_kubernetes_cluster_node_pool" "monr6" {
  name                  = "c4465monr6"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.irmai_aks.id
  vm_size               = "Standard_B4as_v2"
  node_count            = 1
  os_type               = "Linux"
  mode                  = "User"
}

resource "azurerm_kubernetes_cluster_node_pool" "sysff" {
  name                  = "c4465sysff"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.irmai_aks.id
  vm_size               = "Standard_B4as_v2"
  node_count            = 1
  os_type               = "Linux"
  mode                  = "User"
}

# Optional:  Private DNS Zone (Placeholder if you want to manage via TF)
resource "azurerm_private_dns_zone" "blob_dns" {
  name                = "privatelink.blob.core.windows.net"
  resource_group_name = data.azurerm_resource_group.irmai_rg.name
  location            = data.azurerm_resource_group.irmai_rg.location
}
