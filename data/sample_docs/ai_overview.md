# Artificial Intelligence Overview

Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI applications include expert systems, natural language processing, speech recognition, and machine vision.

## Machine Learning

Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future.

### Supervised Learning

Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object and a desired output value. A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.

### Unsupervised Learning

Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels. Unlike supervised learning, unsupervised learning does not have a target variable. Common unsupervised techniques include clustering, association, and dimensionality reduction. Clustering algorithms group data points into clusters based on similarity.

### Reinforcement Learning

Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward. The agent learns to achieve a goal in an uncertain, potentially complex environment. Reinforcement learning differs from supervised learning in not needing labelled input/output pairs.

## Deep Learning

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Deep learning architectures such as deep neural networks, deep belief networks, and recurrent neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and machine translation.

### Neural Networks

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

## Natural Language Processing

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. NLP is used to apply machine learning algorithms to text and speech. Common NLP tasks include text classification, sentiment analysis, named entity recognition, machine translation, and question answering.

## Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with generative models to produce more accurate and grounded answers. RAG first retrieves relevant passages from a knowledge base, then conditions a language model on those passages to generate an answer. RAG reduces hallucination by grounding responses in retrieved evidence and enables updating knowledge without retraining the model.

### RAG Pipeline

A typical RAG pipeline consists of document loading, chunking, embedding, indexing, retrieval, and generation. Documents are split into chunks, embedded into vectors, and stored in a vector database. At query time, the query is embedded and similar chunks are retrieved, then passed to a generator.

### Chunking Strategies

Chunking splits documents into smaller pieces for indexing. Common strategies include fixed-size chunking, recursive character splitting, sentence-based chunking, semantic chunking, and sliding window chunking. Chunk size and overlap affect retrieval quality. Smaller chunks improve precision; larger chunks preserve context.

### Evaluation of RAG

Evaluating RAG systems involves retrieval metrics such as precision, recall, MRR, and nDCG, and generation metrics such as faithfulness, answer relevance, and context precision. Benchmarks include RAGAS and TruLens. Evaluation datasets typically contain questions, ground truth answers, and relevant document ids.

## Knowledge Graphs

A knowledge graph is a structured representation of facts as entities and relations. Knowledge graphs can enhance RAG by providing structured context and multi-hop reasoning. GraphRAG uses community detection and summarization to answer global questions over large corpora.

### GraphRAG

GraphRAG builds a knowledge graph from documents, detects communities, and generates summaries. At query time it uses both local entity neighborhoods and global community summaries to produce answers. GraphRAG excels at questions requiring aggregation across many documents.