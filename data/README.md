# Data Source and Processing Pipeline

This directory documents the data source, processing steps, and artifact management for the Personalized Book Recommender.

No large raw or processed data files are committed to Git. These files are either managed by local download or tracked as versioned  W&B Artifacts.

## Data Source Details

We originally planned to use the complete Amazon Book Reviews dataset, but at about 20 GB it was intractable for smaller AWS computes to preprocess and run training with this dataset. Some experimentation with portions of this dataset can be found the exploratory analysis notebook. The dataset came from Julian McAuley's Amazon Review Dataset, and included ratings, books, and users files in a JSONL format. The link is: https://amazon-reviews-2023.github.io/.


## Smaller Dataset

Due to the size of the dataset above, we decided to use a smaller dataset from Kaggle for testing on smaller computes in AWS. This dataset included book details and reviews CSV files, but only the reviews data was used to train a collaborative filtering model, although it would be worth exploring a hybrid using both datasets and content based filtering as well. The link is: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews. 