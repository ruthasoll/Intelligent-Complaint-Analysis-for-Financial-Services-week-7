# Intelligent-Complaint-Analysis-for-Financial-Services-week-7

# Consumer Complaints Analysis

## Overview
This repository focuses on analyzing consumer complaints from the Consumer Financial Protection Bureau (CFPB) dataset. The goal is to improve customer satisfaction and operational efficiency for CrediTrust by leveraging data-driven insights.

## Table of Contents
- [Business Objective](#business-objective)
- [Task 1: Data Preparation and Initial Exploratory Data Analysis (EDA)](#task-1-data-preparation-and-initial-exploratory-data-analysis-eda)
- [Task 2: Text Chunking, Embedding, and Vector Store Indexing](#task-2-text-chunking-embedding-and-vector-store-indexing)
- [Next Steps](#next-steps)
- [License](#license)

## Business Objective
CrediTrust aims to enhance its consumer complaint management by leveraging sentiment analysis. The primary challenges include:

- **High Complaint Volume**: Identifying root causes of dissatisfaction across product lines.
- **Inefficient Complaint Resolution**: Reducing time and resources spent on resolving complaints.

### Key Performance Indicators (KPIs)
- **Average Resolution Time**: Targeting a 30% reduction in resolution time.
- **Customer Satisfaction Score**: Aiming for a 15% increase in positive feedback.
- **Complaint Volume Reduction**: Reducing the number of complaints by 20%.

To address these challenges, a Retrieval-Augmented Generation (RAG) approach will be implemented to enhance decision-making and customer service.

## Task 1: Data Preparation and Initial Exploratory Data Analysis (EDA)

### Objectives
- Clean and preprocess the CFPB complaint dataset.
- Conduct an initial exploratory data analysis to understand complaint distributions and narrative characteristics.

### Key Steps
1. **Environment Setup**: Fixed SSL issues and established a directory structure for data storage.
2. **Data Download**: Downloaded and extracted the dataset into a Pandas DataFrame.
3. **Exploratory Data Analysis**:
   - Analyzed complaint distributions by product category.
   - Assessed narrative lengths to identify patterns.
4. **Data Cleaning**: Normalized text narratives to enhance quality and reduce noise.
5. **Handling Missing Values**: Addressed significant empty narrative fields.



## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Objectives
- Prepare the clean dataset for semantic analysis by implementing text chunking and generating embeddings.
- Store embeddings in a vector database for efficient retrieval.

### Key Steps
1. **Sampling Strategy**: Employed stratified sampling to select approximately 12,000 complaints, ensuring balanced representation.
2. **Text Chunking**:
   - Used `RecursiveCharacterTextSplitter` to break down narratives into manageable segments (500 characters, 50-character overlap).
3. **Embedding Model Selection**: Chose `sentence-transformers/all-MiniLM-L6-v2` for effective semantic embedding.
4. **Vector Store Integration**:
   - Utilized ChromaDB to persist embeddings with associated metadata for dynamic querying.

---

## Next Steps
- Integrate the Retrieval-Augmented Generation (RAG) pipeline.
- Continue with prompt engineering and user interface development.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
