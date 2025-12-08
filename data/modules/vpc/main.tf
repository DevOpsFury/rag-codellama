resource "aws_vpc" "example" {
  cidr_block = var.cidr_block
  tags = merge(var.tags, {
    Name = "example-vpc"
  })
}
