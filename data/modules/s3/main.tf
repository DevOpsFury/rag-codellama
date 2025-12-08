resource "aws_s3_bucket" "example" {
  bucket = var.bucket_name
  tags = merge(var.tags, {
    Name = "example-s3"
  })
}
