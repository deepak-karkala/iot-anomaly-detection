resource "aws_ecr_repository" "ad_training_repo" {
  name                 = local.ecr_repo_name_full
  image_tag_mutability = "MUTABLE" # Or IMMUTABLE for stricter versioning

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}