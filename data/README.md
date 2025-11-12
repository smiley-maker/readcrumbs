# Data Source and Processing Pipeline

This directory documents the data source, processing steps, and artifact management for the Personalized Book Recommender.

No large raw or processed data files are committed to Git. These files are either managed by local download or tracked as versioned  W&B Artifacts.

## Data Source Details

The foundation of our recommendation model is the Amazon Review Data.

| Attribute | Details |
| :------- | :------: |
| Dataset Name | Amazon Review Data — Books Subset |
| Original Source     | Julian McAuley's Amazon Review Dataset   |
| Dataset Components   | Ratings, Books, and Users files (specific format depends on chosen subset)   |
| Size     | 10.3 million users, 4.4 million items, and 29.5 million ratings     |
| License     | Open access for non-commercial research purposes.     |
| Link     | https://amazon-reviews-2023.github.io/     |