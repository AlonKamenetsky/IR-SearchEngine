# Search Engine Backend (Inverted Index + Flask Frontend on GCP)

## Overview
This project implements a simple search engine backend over a large Wikipedia
corpus. The system is composed of two main parts:

1. **Offline indexing** – building and storing an inverted index efficiently
   using disk and Google Cloud Storage (GCS).
2. **Online search service** – a Flask-based REST API that loads the index and
   answers search queries using BM25 ranking.

The system is deployed on **Google Cloud Platform (GCP)** and is designed to
work with very large posting lists that cannot fit entirely in memory.

---

## Project Structure

.

├── inverted_index_gcp.py # Core inverted index implementation 

├── search_frontend.py # Flask REST API for search

├── startup_script_gcp.sh # VM initialization (Python + dependencies)

├── run_frontend_in_gcp.sh # GCP deployment & execution script

├── run_frontend_in_colab.ipynb # Colab helper notebook

└── README.md


---

## Core Components

## 1. Inverted Index (`inverted_index_gcp.py`)
This module contains the core data structures and logic for building, storing,
and reading the inverted index from a GCP storage bucket. This file was given in the intructions of the project.

---

## 2. Search Frontend (`search_frontend.py`)
This file implements the online search service using Flask.

### Responsibilities
- Load the inverted index from Google Cloud Storage
- Tokenize incoming queries
- Score documents using BM25
- Serve results via REST API endpoints

---

### Index Loading
The index is loaded once when the server starts:
- Global statistics are loaded from `index.pkl`
- Posting lists are fetched lazily from GCS during query processing

This design minimizes startup time and memory usage.

---

### Query Processing Flow
1. Receive query via HTTP request
2. Tokenize query using a regex-based tokenizer
3. For each query term:
   - Read its posting list from disk/GCS
   - Compute its BM25 contribution
4. Aggregate scores across documents
5. Sort and return the top 100 results

BM25 parameters:
- `k1 = 1.5`
- `b = 0.75` (document length normalization omitted for simplicity)

---

### REST API Endpoints

| Endpoint | Description |
|--------|------------|
| `/search` | BM25-based search (top 100 results) | 
| `/search_body` | Placeholder for TF-IDF + cosine similarity |
| `/search_title` | Placeholder for title-based search |
| `/search_anchor` | Placeholder for anchor-text search |
| `/get_pagerank` | Placeholder for PageRank values |
| `/get_pageview` | Placeholder for pageview statistics |

Only the `/search` endpoint is fully implemented. Other endpoints are included
as stubs for future extensions.

---

## 3. VM Startup Script (`startup_script_gcp.sh`)
This script prepares a fresh Google Compute Engine VM.

### What it does
- Installs Python, pip, and virtual environment tools
- Creates a Python virtual environment
- Installs required dependencies:
  - Flask
  - Werkzeug
  - NumPy
  - Pandas
  - Google Cloud Storage SDK
  - NLTK

The script runs automatically when the VM is created.

---

## 4. GCP Deployment Script (`run_frontend_in_gcp.sh`)
Automates deployment of the search frontend on GCP.

### Main steps
1. Allocate a static external IP address
2. Open firewall access to port `8080`
3. Create a Compute Engine VM instance
4. Copy the Flask application to the VM
5. Start the server in the background
6. Test the service using `curl`

Cleanup commands are included to prevent unnecessary cloud charges.

---

## 5. Colab Helper Notebook (`run_frontend_in_colab.ipynb`)
A helper notebook for running the frontend inside Google Colab using `ngrok`.
Useful for testing without provisioning a GCP VM.

---

## Design Considerations
- **Scalability**: Posting lists stored on disk or GCS
- **Efficiency**: Compact binary encoding and fixed-size files
- **Modularity**: Clear separation between indexing and querying
- **Cloud-ready**: Designed for seamless deployment on GCP

---

## Limitations and Future Work
- BM25 does not include document length normalization
- Several search endpoints are placeholders
- No caching for frequent queries
- No stemming or query expansion

---

## Summary
This project demonstrates a complete search engine backend including:
- Large-scale inverted index storage
- Efficient disk-based retrieval
- Cloud deployment on GCP
- RESTful search APIs

The design follows real-world information retrieval principles while remaining
simple, efficient, and extensible.
