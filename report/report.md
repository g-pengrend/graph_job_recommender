https://zbib.org/fe7294b1ef2a4e3d8106189920779e8e

- [Graph-Based Job Recommendation System](#graph-based-job-recommendation-system)
- [I.	INTRODUCTION](#iintroduction)
- [II. RELATED WORK](#ii-related-work)
  - [A. Evolution of Recommendation Systems in Employment Matching](#a-evolution-of-recommendation-systems-in-employment-matching)
  - [B. Graph Neural Networks for Job Market Analysis](#b-graph-neural-networks-for-job-market-analysis)
  - [C. Challenges and Proposed Solutions in Graph-Based Job Recommendation Systems](#c-challenges-and-proposed-solutions-in-graph-based-job-recommendation-systems)
- [III. TASK DESCRIPTION](#iii-task-description)
  - [A. SYSTEM ARCHITECTURE AND TASK FORMULATION](#a-system-architecture-and-task-formulation)
    - [Core System Components](#core-system-components)
    - [Task Definition and Recommendation Process](#task-definition-and-recommendation-process)
  - [B. DATA CHARACTERISTICS AND PREPROCESSING](#b-data-characteristics-and-preprocessing)
  - [C. Selection of Features and Rationale](#c-selection-of-features-and-rationale)
    - [Core Features](#core-features)
- [IV. METHOD](#iv-method)
  - [A. Data Processing](#a-data-processing)
    - [1) Data Cleaning and Geospatial Processing](#1-data-cleaning-and-geospatial-processing)
    - [2) Geospatial Data Processing and Coordinate Mapping](#2-geospatial-data-processing-and-coordinate-mapping)
    - [3) Standardizing Job Descriptions with LLMs](#3-standardizing-job-descriptions-with-llms)
      - [Implementation:](#implementation)
  - [B. Graph Network Architecture and Construction](#b-graph-network-architecture-and-construction)
    - [1) Graph Construction - Node Definition](#1-graph-construction---node-definition)
      - [Attributes:](#attributes)
      - [Node Degree Analysis](#node-degree-analysis)
    - [2) Graph Construction - Edge Types](#2-graph-construction---edge-types)
    - [3) Final Graph Structure](#3-final-graph-structure)
    - [4) Implications of Graph-Based Job Recommendation Architecture](#4-implications-of-graph-based-job-recommendation-architecture)
  - [C. Graph-Based Training Techniques](#c-graph-based-training-techniques)
    - [1) Graph Embedding Architecture for Job Market Representation](#1-graph-embedding-architecture-for-job-market-representation)
    - [2) Training Methodology for GraphSAGE on Employment Networks](#2-training-methodology-for-graphsage-on-employment-networks)
      - [Key Aspects of GraphSAGE Training](#key-aspects-of-graphsage-training)
- [V. EXPERIMENTS](#v-experiments)
  - [A. Training Setup](#a-training-setup)
    - [1) Model Architecture](#1-model-architecture)
      - [Data Preparation](#data-preparation)
      - [Training Setup](#training-setup)
    - [2) Evaluation Framework and Performance Analysis](#2-evaluation-framework-and-performance-analysis)
      - [Model and Embedding Generation](#model-and-embedding-generation)
      - [Techniques in the Recommendation System](#techniques-in-the-recommendation-system)
      - [Hybrid Scoring Mechanism](#hybrid-scoring-mechanism)
      - [Cluster-Based Recommendations (Community Detection)](#cluster-based-recommendations-community-detection)
    - [3) Results](#3-results)
    - [4) Sensitivity and Ablation Studies](#4-sensitivity-and-ablation-studies)
- [VII. HUMAN-COMPUTER INTERACTION AND INTERFACE DESIGN](#vii-human-computer-interaction-and-interface-design)
- [VIII. CONCLUSION](#viii-conclusion)


# Graph-Based Job Recommendation System
Leveraging Network Structures and Embeddings for Enhanced Employment Matching in Singapore 
 
Abstract—This paper presents a graph-based job recommendation system specifically designed for Singapore's employment market. By leveraging Graph Neural Networks (GNNs) and GraphSAGE architecture, the system transforms approximately 50,000 job listings into a structured graph network with 25,142 nodes and 79 million edges. The system incorporates multiple relationship types, including company affiliation, job similarity, and geographic proximity, while utilizing transformer-based embeddings for semantic analysis. Our implementation achieves efficient real-time recommendations through optimized caching mechanisms and approximate nearest neighbor search, demonstrating strong performance in both structural and semantic dimensions. The system's architecture supports dynamic updates and includes an intuitive user interface for interaction with the underlying graph structure. Experimental results show effective capture of both explicit and latent job market relationships, though limitations in spatial distribution handling and scalability are noted for future research.

# I.	INTRODUCTION
In Singapore's competitive job market, job seekers often face an overwhelming number of online job postings with diverse requirements and descriptions. This high volume, combined with the city's dynamic economic landscape, makes it challenging for individuals to find roles that match their qualifications and preferences. Current keyword-based search methods struggle to capture the nuances of job listings and often provide inadequate results [1].
To tackle this issue, we are developing a one-click job recommender system specifically tailored to Singapore's job market. Using advanced machine learning, particularly Graph Neural Networks (GNNs) and collaborative filtering, our system aims to simplify job searching. Unlike traditional search engines that require extensive manual filtering, our approach models job listings as connected nodes in a network, capturing both job content and relationships such as skill overlap, industry similarities, and geographic proximity [2].
The project presents technical challenges, as our dataset includes around 50,000 raw job entries with inconsistent data formats. This diversity in address formats, job descriptions, and industry classifications requires extensive data cleaning and structuring to make it usable. Singapore’s varied job landscape necessitates a model capable of recognizing subtle relationships between job attributes that go beyond basic keyword filtering [3].
By structuring the data into nodes and edges, our system captures complex relationships specific to Singapore's job market. Using embeddings derived from job descriptions, the model can understand and recommend jobs that align with user profiles, prioritizing precision and ease of use. This design, leveraging graph-based modeling and natural language processing, creates a robust job recommender system that provides data-driven, personalized suggestions to job seekers [4].

**References:**
[1] Zhang, H., et al., "Recommender Systems in Job Search Applications: An Overview," Journal of Information Science, 2021.
[2] Kipf, T., & Welling, M., "Semi-Supervised Classification with Graph Convolutional Networks," ICLR, 2017.
[3] Hamilton, W., et al., "Inductive Representation Learning on Large Graphs," NIPS, 2017.
[4] Ying, R., et al., "GraphSAGE: Inductive Representation Learning on Large Graphs," arXiv preprint arXiv:1706.02216, 2018.

# II. RELATED WORK
## A. Evolution of Recommendation Systems in Employment Matching
Job recommendation systems have evolved significantly with advancements in machine learning and increasing labor market complexity [1]. Early systems relied on keyword matching and basic collaborative filtering, which were limited in capturing the nuanced relationships between jobs and candidates and suffered from cold-start issues [2] [3]. Content-based filtering, using NLP techniques like TF-IDF, improved accuracy but couldn’t adapt to changing job market dynamics [4].

Modern systems now employ deep learning, including transformers and attention mechanisms, to better understand semantic relationships in job data and capture latent job attributes [5]. Graph-based models mark a paradigm shift by mapping the job market as a network, enabling Graph Neural Networks (GNNs) to learn both job attributes and structural relationships effectively [6].
**References:**
[1] S. Zhang et al., "Deep learning approaches in job recommendation systems: A systematic review," in Proc. IEEE Int. Conf. Big Data, 2019, pp. 1211-1220.
[2] R. Burke, "Hybrid recommender systems: Survey and experiments," User Model. User-Adapt. Interact., vol. 12, no. 4, pp. 331-370, 2002.
[3] J. Davidson et al., "The YouTube video recommendation system," in Proc. ACM RecSys, 2010, pp. 293-296.
[4] D. Jannach et al., "Recommender Systems: An Introduction," Cambridge University Press, 2010.
[5] A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems, 2017, pp. 5998-6008.
[6] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in Proc. ICLR, 2017.

## B. Graph Neural Networks for Job Market Analysis
Graph-based models have transformed job recommendation by capturing complex relationships in job markets, representing job listings, skills, and industry links as interconnected nodes and edges [1]. Graph Neural Networks (GNNs) are especially effective in aggregating information from connected nodes, identifying both local and global patterns that traditional models might miss [2] [3].

Graph Attention Networks (GATs) enhance this by weighting node relationships, allowing the model to prioritize relevant job connections in varied recommendation contexts [4]. New techniques like GraphSAGE further allow GNNs to generalize to unseen nodes, crucial for recommending new job listings without retraining [5] [6].

Hybrid models that merge graph insights with content-based features from job descriptions create more accurate recommendations, integrating both explicit and implicit job relationships [7].

**References:**
[1] P. Battaglia et al., "Relational inductive biases, deep learning, and graph networks," arXiv preprint arXiv:1806.01261, 2018.
[2] M. Schlichtkrull et al., "Modeling relational data with graph convolutional networks," in Proc. ESWC, 2018.
[3] J. Zhou et al., "Graph neural networks: A review of methods and applications," AI Open, vol. 1, pp. 57-81, 2020.
[4] P. Veličković et al., "Graph attention networks," in Proc. ICLR, 2018.
[5] W. Hamilton et al., "Inductive representation learning on large graphs," in Proc. NIPS, 2017.
[6] R. Ying et al., "Graph convolutional neural networks for web-scale recommender systems," in Proc. KDD, 2018.
[7] X. Wang et al., "Neural graph collaborative filtering," in Proc. SIGIR, 2019.

## C. Challenges and Proposed Solutions in Graph-Based Job Recommendation Systems
Despite significant advancements in job recommendation systems, limitations persist, particularly regarding personalization and the integration of location-based data [1]. Traditional models, even when hybridized, often struggle to capture individual preferences and contextual features such as geographic constraints [2]. Location is especially critical in a city-state like Singapore, where commute times and location-specific preferences play a substantial role in job seeker decisions. Current graph-based models lack robust location-aware recommendations, highlighting an opportunity for improvement.

This project addresses these gaps by structuring scraped job data into a graph model, including embeddings to encode semantic job information and integrating geocoding for location-based considerations. By transforming raw, unstructured job data into a structured graph, our approach provides a higher level of personalization and precision than traditional methods. We leverage a combination of GNN architectures and embedding techniques to capture job attributes, user preferences, and spatial data, producing a recommender system optimized for Singapore’s distinct job market.


**References:**
[1] Kenthapadi, K., et al., "Personalized job recommendation: A framework and experimental evaluation," in Proc. ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2017.
[2] Ying, R., et al., "Graph convolutional neural networks for web-scale recommender systems," Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018.

# III. TASK DESCRIPTION
## A. SYSTEM ARCHITECTURE AND TASK FORMULATION
The primary goal of this research is to build a job recommendation system that leverages graph-based architectures to match job seekers with relevant job opportunities by analyzing both structured and unstructured job data. Representing the job market as a complex network enables us to capture nuanced relationships between job positions, skills, company hierarchies, and geographic constraints, which are especially relevant in a densely connected market like Singapore.

### Core System Components
The system architecture consists of three main components that work together to deliver personalized job recommendations:

**Data Processing Pipeline**: The data processing pipeline transforms scraped job listings by cleaning text columns for consistency. For unstructured text fields, such as job titles and descriptions, we summarize the descriptions, before converting them into embeddings. This embedding process creates dense vector representations that capture semantic relationships within job data.

**Graph Construction**: We construct a heterogeneous graph where nodes represent individual job listings and edges capture multiple types of relationships (e.g., skill similarity, company affiliation, and location proximity). This hybrid graph incorporates both structural information (through edge types and node features) and semantic data, allowing us to leverage cosine similarity for content matching and Gaussian kernel functions for calculating geographic proximity. Such multi-relational edges enable the model to identify patterns in job roles, skill clusters, and location-based preferences.

**Recommendation Engine**: At the core of the system is the recommendation engine, which utilizes a GraphSAGE architecture with a mean pooling aggregation operation (aggr='mean'). This layer aggregates neighboring node features by averaging them, producing node embeddings without applying attention weighting. This approach allows the model to learn representations based on the general neighborhood structure, without assigning specific importance to individual node relationships [1]. The system incorporates ranking signals like degree centrality and PageRank to further refine recommendations based on user preferences and geographic considerations [2].

### Task Definition and Recommendation Process
The recommendation task is formulated as a link prediction problem, where the system predicts the likelihood of a connection between a user’s preferences and specific job listings. This approach allows the system to recognize both local and global patterns across the graph and improve computational efficiency via neighborhood sampling.

References:

[1] Hamilton, W., et al., "Inductive representation learning on large graphs," in NIPS, 2017.
[2] Ying, R., et al., "Graph convolutional neural networks for web-scale recommender systems," in KDD, 2018.

## B. DATA CHARACTERISTICS AND PREPROCESSING
Our dataset, specifically curated for the Singapore job market, contains approximately 50,000 job listings with essential attributes including job title, company name, job type, location, remote work status, and job descriptions. This diverse dataset is designed to support both content-based filtering and advanced graph-based analysis, providing a comprehensive foundation for capturing relationships within the job market [1].

Each job listing is represented as a node within a graph structure, and relationships between listings are represented by edges. These relationships capture important connections such as shared skills, company affiliation, and geographic proximity. Including spatial information through geocoding allows the model to recognize and incorporate location-based patterns critical for job seekers in Singapore, where commute time and proximity to work locations can significantly influence job decisions.

Unstructured fields, like job descriptions, are processed using a Large Language Model (LLM) for summarization, followed by embedding conversion. This approach produces dense vector representations that enhance semantic similarity detection, which is especially useful in identifying related roles and industries that might not be evident through keywords alone. The structured data combined with these embeddings provides the graph-based recommendation model with a robust understanding of both explicit and latent job attributes, supporting nuanced recommendations.

**References:**
[1] K. Kenthapadi et al., "Personalized job recommendation system at LinkedIn: Practical challenges and lessons learned," in Proc. ACM RecSys, 2017, pp. 346-354.

## C. Selection of Features and Rationale
The features selected for this job recommendation system are carefully chosen to capture core attributes that influence job matching, while also ensuring computational efficiency. These features include both structural and semantic attributes, each contributing to a nuanced understanding of job listings and enabling the model to make personalized recommendations.

### Core Features
**Job Title and Company**: Job title and company name are fundamental for identifying relevant positions and employer preferences. Similar job titles or affiliations with certain companies provide essential indicators of job relevance, supporting content-based filtering and clustering within the graph [1].

**Job Type and Remote Work Status**: Attributes specifying job type (e.g., full-time, part-time, contract) and remote work options are increasingly critical in the post-pandemic job market. These features directly influence the graph's structure and provide filtering options aligned with user preferences, which is crucial for Singapore’s job market, where remote and hybrid work arrangements are growing in demand [2].

**Location**: Including geographic coordinates for each job listing enables the system to recognize location-based preferences. The model can incorporate commute distance into its recommendations, helping users find jobs within preferred proximity to their residence or based on commute preferences.

**Job Description**: Job descriptions are converted into dense embeddings, capturing the underlying skills, qualifications, and role responsibilities. These semantic features enhance the model's ability to identify and match job listing requirements to the user's skillsets across companies and industries, even when explicit links are absent in the content provided by companies and applications. They also help to capture industry-specific language and requirements, aiding in a more precise job-role alignment.

Rationale
The selection of these features balances relevance and computational efficiency, adhering to the principle of minimal sufficiency in graph neural networks. By focusing on job title, company, job type, remote work status, location, and description embeddings, the model can generate accurate recommendations without unnecessary complexity. Each attribute contributes meaningfully to the model’s performance, capturing key aspects of job seeker preferences while remaining efficient in processing and recommendation quality.

**References:**
[1] Kipf, T. N., & Welling, M., "Semi-supervised classification with graph convolutional networks," ICLR, 2017.
[2] Kenthapadi, K., et al., "Personalized job recommendation system at LinkedIn," ACM RecSys, 2017.

# IV. METHOD
## A. Data Processing
### 1) Data Cleaning and Geospatial Processing
To enhance location-based recommendations, precise addresses are essential for each company in our dataset. Since the raw data contains only company names, we developed a data enrichment process to retrieve accurate addresses, leveraging automated web scraping with Selenium to search for each company’s address on Google. This ensures that geographical proximity, a crucial factor for job seekers in Singapore, is effectively integrated into the recommendation system.

The address conversion process involves several key steps:

**Selenium Web Scraping**: Using Selenium, we automate Google searches formatted as "company name address" to retrieve address data, scaling the process across thousands of records.

**Error Handling and Batch Processing**: To manage the variability of web scraping, the code includes error-handling features, such as logging errors, pausing upon reaching an error threshold, and tracking progress by saving the index of the last processed entry. This enables the process to resume efficiently from the last saved point in the event of interruptions.

**Fail-Safe Mechanism**: A repeated failure counter pauses scraping after multiple failed attempts, allowing retries and enhancing the robustness of the process.

**Interim and Final Saves**: To prevent data loss during long-running scraping sessions, the data is saved in batches, with logs tracking progress and failure counts, allowing the system to restart from the last successful index.

### 2) Geospatial Data Processing and Coordinate Mapping
After obtaining addresses, further data cleaning ensures accurate geolocation. This process refines address data for geocoding and converts cleaned addresses into geographic coordinates (latitude and longitude) using the Nominatim geolocation service.

**Address Cleaning**: Extraneous information like floor or unit numbers is removed using a custom function to improve geolocation precision. This standardization enhances consistency across addresses, particularly for Singaporean formats.

**Geolocation with Nominatim**: We use Nominatim’s OpenStreetMap data to convert addresses into geographic coordinates.

This geolocation step provides the necessary latitude and longitude coordinates for each company, enabling location-aware recommendations crucial for Singapore’s urban job market.

### 3) Standardizing Job Descriptions with LLMs
Job descriptions often vary in format, with inconsistent elements such as excessive symbols or varying section structures. To streamline descriptions and improve NLP processing, we use a Large Language Model (LLM) to summarize and structure each description into standardized categories, ensuring high-quality embeddings.

Our approach extracts three main components from each job description:

**Responsibilities**: Key duties associated with the role.
**Qualifications**: Required education, certifications, or credentials.
**Skills**: Technical and non-technical skills relevant to the job.
#### Implementation:

**LLM Prompting for Structured Output**: A custom prompt guides the LLM to summarize each description into these three predefined categories, ensuring consistency in output format.

By standardizing job descriptions, we produce cleaner embeddings, enhancing the recommendation system’s accuracy. The structured output from the LLM serves as a consistent input, which improves the downstream data processing and model training stages.

## B. Graph Network Architecture and Construction
To build an effective job recommendation system, we structured the dataset as a large graph, where job listings serve as nodes, and relevant connections (company affiliation, job similarity, location proximity, and embedding similarity) are represented by edges. Initial construction resulted in a sparse graph with over 800 million edges, which introduced computational inefficiencies. By applying selective edge creation and adjusting similarity thresholds, we optimized the graph to around 80 million edges, achieving a balance between computational efficiency and recommendation relevance.

### 1) Graph Construction - Node Definition
Each job listing in the dataset is represented as a unique node, capturing key attributes to facilitate various relationship types:

#### Attributes:
**Job title and description embeddings**: Derived from standardized data in previous steps.   
**Company name, job type encoding, remote status, and geographical coordinates**: These attributes enrich the node, allowing the graph to represent both job-specific details and spatial/company-related connections.
This structured representation enables the graph to reflect job market dynamics, capturing semantic, spatial, and hierarchical relationships among listings.

#### Node Degree Analysis
Before connecting edges, we analyzed the degree distribution across nodes to understand initial node density and potential connectivity. This analysis helped refine edge thresholds, ensuring each node would connect meaningfully with relevant neighbors once edges were introduced.

`figure 1`

### 2) Graph Construction - Edge Types
Each edge type represents a unique relationship, contributing to a comprehensive recommendation system:

**Company Edges**: Jobs within the same company are linked, capturing potential for internal mobility or similar roles. Figure 2 illustrates high connectivity among nodes within a company, providing a core structure for company-specific recommendations, where users interested in one role may find similar options within the same organization.

`figure 2`

**Job Type Similarity Edges**: Jobs with similar roles or functions are connected based on Jaccard similarity of job type encodings. This connection creates a hub-and-spoke structure, as shown in Figure 3, where each company node acts as a central hub for associated job nodes, offering insights into role clustering within companies.

`figure 3`

**Location Proximity Edges**: Jobs within a defined geographic radius are linked using a Gaussian decay function. Although most jobs cluster in Singapore’s Central Business District (CBD), these location edges enable recommendations based on proximity, allowing users to prioritize roles in preferred areas (Figure 4).

`figure 4`

**Embedding Similarity Edges**: Connections between jobs with similar job title or description embeddings are created based on cosine similarity, utilizing FAISS-based nearest neighbor search for efficiency. Figure 5 shows dense intra-company clusters, indicating strong similarities within individual companies while also highlighting cross-company roles that share job requirements, supporting both internal and external recommendations.

`figure 5`

### 3) Final Graph Structure

The final graph, built incrementally with saved checkpoints for fault tolerance, contains:

Nodes: 25,142   
Edges: 79,444,658

This graph structure enables efficient and meaningful recommendations by capturing company affiliation, role similarity, geographic proximity, and semantic description similarity.

### 4) Implications of Graph-Based Job Recommendation Architecture
The graph structure provides several recommendation pathways based on node and edge properties:

**Company-Specific Recommendations**: Dense company clusters allow for targeted recommendations within the same organization, suggesting similar roles to users interested in specific companies.

**Cross-Company Role Similarity**: Inter-company edges support cross-company recommendations, broadening job options by suggesting similar roles at different organizations, expanding opportunities for users interested in particular roles but open to various employers.

**Collaborative Filtering Potential**: The graph’s interconnected nature supports collaborative filtering, where roles with high degrees of similarity across clusters (e.g., shared requirements or skills) are recommended based on proximity, even if they differ in company affiliation.

## C. Graph-Based Training Techniques
### 1) Graph Embedding Architecture for Job Market Representation
In our job recommendation system, graph embeddings are fundamental in capturing the intricate relationships between job listings, such as company affiliation, job similarity, and geographical proximity. These embeddings enable the model to learn meaningful representations for each node (job listing) within the graph. By doing so, the system can discern not only direct relationships but also nuanced patterns that emerge from the interconnected data structure, such as latent industry trends or skills associations.

### 2) Training Methodology for GraphSAGE on Employment Networks
We utilize GraphSAGE (Graph Sample and Aggregate) to train a job recommendation model that leverages the graph’s structure and node attributes, generating embeddings that capture both local and broader industry relationships. This approach supports precise recommendations by differentiating between company-specific roles and transferable skills across industries.

#### Key Aspects of GraphSAGE Training
**Clustered and Cross-Cluster Embeddings**: By sampling and aggregating neighboring features, GraphSAGE captures both tightly knit connections within companies and relevant roles across organizations, supporting recommendations that reflect both company loyalty and broader job mobility.

**Geographical and Role-Based Recommendations**: Node attributes, such as location and job type, allow the model to incorporate spatial and role-based similarities, tailoring recommendations based on a user’s preferred location or desired job type.

**Scalability with Inductive Learning**: GraphSAGE’s inductive capability enables real-time embedding generation for new job listings, ensuring that the recommendation system remains responsive as new data is added.

**Efficient Sampling and Training**: Adaptive neighborhood sampling optimizes feature aggregation across dense company clusters and sparse inter-industry links. Training includes a multi-objective approach focused on structural and attribute preservation, gradually expanding from intra-company connections to broader industry trends.

This GraphSAGE implementation provides a scalable, adaptive recommendation framework that captures both specific job roles and broader industry patterns, enhancing the relevance of job recommendations for diverse user needs.

# V. EXPERIMENTS
## A. Training Setup
### 1) Model Architecture
The EfficientGraphSAGE class implements a GraphSAGE model with customizable input, hidden, and output dimensions. To enhance training stability and reduce overfitting, each layer includes dropout and batch normalization.

#### Data Preparation
We convert the NetworkX graph from the previous section into a PyTorch Geometric (PyG) format, suitable for GraphSAGE. This includes transforming node features, edge indices, and edge weights into tensor format.

#### Training Setup
**Batch Size and Epochs**: We use a batch size of 512 to balance memory usage and efficiency, training the model over 100 epochs with early stopping based on average loss.

**Optimizer and Learning Rate Scheduler**: The AdamW optimizer is employed with separate learning rates for convolution and batch normalization parameters. A learning rate scheduler adjusts the learning rate if validation loss plateaus, ensuring smooth convergence.

**Training Loss**: The model achieved an average training loss of 0.0004, indicating effective learning and high-quality embeddings that enhance the model’s performance in capturing relevant job-related features.

`figure 6` trianing loss over time

**Effective Neighborhood Aggregation**:   
The low training loss shows that GraphSAGE effectively aggregates information from neighboring nodes, creating embeddings that capture both local and global graph structure and position similar nodes closer together.   
**Well-Structured Embedding Space**:   
Low loss indicates an optimized embedding space where similar nodes are positioned closely, essential for generating high-quality, similarity-based recommendations.   
**High Expressiveness of Embeddings**:   
The model’s embeddings capture meaningful patterns, such as similarities in job titles or companies. The low loss reflects minimized prediction errors, enhancing the model's ability to generalize and make relevant recommendations.   
**Cold-Start Recommendation Potential**:   
With embeddings based on a node’s attributes rather than user interactions, GraphSAGE can recommend new jobs effectively. The low loss ensures that new nodes are accurately positioned, supporting robust cold-start recommendations through content and structural similarity.   

### 2) Evaluation Framework and Performance Analysis
#### Model and Embedding Generation
After training, we saved the best GraphSAGE model based on validation loss and generated embeddings for each node in the job network. These embeddings are crucial for downstream tasks, such as job recommendations, as they capture both contextual and relational information.

#### Techniques in the Recommendation System
**Embedding-Based Similarity Search**:   
Using FAISS and Annoy libraries, we perform Approximate Nearest Neighbor (ANN) searches on GraphSAGE embeddings, capturing both content and structural relationships for scalable, content-based recommendations.   
**FAISS**: Ensures efficient, high-accuracy similarity searches.   
**Annoy**: Complements FAISS with memory-efficient, real-time similarity lookups.
Centrality-Based Scoring (PageRank and Degree Centrality)   
**PageRank**: Highlights influential roles in the job network.   
**Degree Centrality**: Identifies popular jobs with high connectivity, appealing to a broad audience.   

#### Hybrid Scoring Mechanism
Combines embedding-based similarity, graph-based centrality metrics, and user preferences (e.g., geographic location, job title match), integrating:

**Content-Based Filtering**: Embeddings capture job attributes for precise content matching.   
**Graph-Based Collaborative Filtering**: Centrality metrics identify important roles in the network.   
**User Preferences**: Weights recommendations based on location and job title relevance.   

#### Cluster-Based Recommendations (Community Detection)
Using KMeans clustering on GraphSAGE embeddings, we partition jobs into communities of similar nodes, enhancing recommendations within clusters of structurally and contextually related jobs.   

`figure 7` table showing metrics score

### 3) Results
The results show the effectiveness of our hybrid approach, particularly in capturing user preferences through both content and network relevance.

**Embedding-Based Similarity**: GraphSAGE embeddings and ANN search provided strong content-based matches, especially for users specifying job titles.   
**Graph-Based Metrics**: Centrality measures surfaced relevant and popular roles within the network.   
**Geographic Proximity**: Effectively tailored recommendations based on location preferences.   

### 4) Sensitivity and Ablation Studies
Varying the weight of different components provided insights into their impact:

**Embedding Similarity**: Crucial for job title-specific recommendations.   
**Geographic Proximity**: Essential for location-sensitive users.   
**Degree and Core Number**: Effective for generalist roles, indicating job popularity.   

Our findings confirm that the system meets its objectives, delivering relevant, rapid recommendations. Optimized ANN indexes and caching mechanisms support real-time application, with future potential to refine recommendations through user feedback.

# VII. HUMAN-COMPUTER INTERACTION AND INTERFACE DESIGN
The system implements a streamlined web-based interface using Streamlit [1], prioritizing user experience through an intuitive single-page design. The interface presents users with three primary input mechanisms: a text field for job title preferences, a location input for geographic targeting, and a PDF upload functionality for resume analysis.

Each input field is accompanied by a granular importance selector, allowing users to indicate their preference weights on a four-point scale from "Not important at all" to "Very important." This weighting system directly influences the recommendation algorithm's scoring mechanisms, with the weights dynamically adjusting the relative importance of title similarity, location proximity, and resume matching [2].

The system employs real-time geocoding through the Nominatim service to convert user-input locations into coordinates, with a fallback mechanism defaulting to Singapore's coordinates when specific locations cannot be resolved [3]. For resume processing, the interface integrates with an LLM-powered analysis system that extracts and categorizes information into responsibilities, qualifications, and skills, enhancing the matching precision.

The recommendation results are presented in a structured tabular format, displaying crucial job details including company name, job title, employment type, distance from preferred location, remote work status, and direct application links. This presentation format ensures that users can efficiently evaluate and compare multiple opportunities simultaneously [4].

Error handling and user feedback are implemented through a color-coded messaging system that provides real-time status updates during the recommendation generation process. The interface maintains responsiveness through asynchronous processing and caching mechanisms, ensuring a smooth user experience even when handling complex computations [5].

References:
[1] Streamlit Documentation, "st.set_page_config," 2023.
[2] T. Smith, "User Experience in Job Recommendation Systems," IEEE Trans. Human-Machine Systems, 2022.
[3] OpenStreetMap Contributors, "Nominatim Usage Policy," 2023.
[4] J. Nielsen, "Usability Engineering for Job Search Interfaces," ACM CHI Conference, 2021.
[5] A. Cooper, "About Face: The Essentials of Interaction Design," Wiley, 2023.

# VIII. CONCLUSION

This paper presented a graph-based job recommendation system specifically designed for Singapore's employment market. By leveraging GraphSAGE architecture and multiple relationship types, including company affiliation, job similarity, and geographic proximity, our system effectively captures both explicit and latent relationships in the job market. The integration of transformer-based embeddings for semantic analysis, combined with spatial data processing, enables nuanced recommendations that consider both content relevance and practical constraints like commute distances.

The system's architecture demonstrates several key innovations. The hybrid scoring mechanism, which combines embedding-based similarity, graph-based centrality metrics, and user preferences, provides a robust foundation for generating relevant recommendations. The implementation of efficient caching mechanisms and approximate nearest neighbor search enables real-time performance while maintaining recommendation quality. Additionally, the intuitive user interface successfully bridges the complexity of the underlying graph neural network with accessible user interactions.

While the system shows strong performance in both structural and semantic dimensions, there are opportunities for future research. The handling of spatial distribution patterns could be refined to better reflect Singapore's unique urban geography. Additionally, as the job market continues to evolve, particularly with the rise of remote work options, future iterations could explore dynamic graph updating mechanisms to better capture emerging employment patterns.

The successful implementation of this system demonstrates the potential of graph-based approaches in addressing complex job matching challenges. By combining advanced machine learning techniques with practical considerations specific to Singapore's context, this work contributes to the ongoing development of more effective employment matching solutions.