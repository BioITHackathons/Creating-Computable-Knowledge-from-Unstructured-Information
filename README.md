# Creating-Computable-Knowledge-from-Unstructured-Information
## Purpose and Goals 

### Purpose
This repository is dedicated to the 2022 BioIT HAckathon. This team was tasked with creating and optimizing a natural language processing (NLP) pipline derived from NVIDIA's MEGATRON pipleine. The models developed in this repository will be utilized to analyze abstracts from scientific publications in the PubMed repository in order to create a knowledge graph linking disease to key genes, proteins, and drug interactions 

### Goals 
The goals of this project are to identifiy a biology focused training data for NLP and use it to train a deep learning model for disease- drug, disease-gene, and disease-protein interactions. Information gleaned from this model were then visualized in a knowledge graph. We use the standard Name Entity REcognition (NER), Reletive Extraction (RE), followed up Entity Linking (EL) pipeline commonly used by NVIDIA's Megatron. We compared the accuracy of this pipeline with a p-tunining and prompt tuning pipeline also within Megatron. 

## Experimental Setup

### Training Set Data
We utilized the BioCreative data sets to train our models. 
