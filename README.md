# 🧠 Build LLMs From Scratch

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/5/5a/Transformer_model.png" width="600">
</p>

This repository contains my learning journey and implementations from the **“Building LLMs from Scratch”** series (Vizuara).  
It covers everything from **fundamentals of NLP → transformer internals → training → scaling → applications like chatbots & fine-tuning**.

---

## 📚 Topics Covered

### **1. Introduction & Basics**
- What are Large Language Models (LLMs)?
- History of NLP → RNN → LSTM → Transformers
- Why LLMs became state-of-the-art

---

### **2. Data & Tokenization**
- Text preprocessing
- Tokenization (WordPiece, BPE)
- Vocabulary building
- Implementing a custom tokenizer

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/89/Byte_pair_encoding.svg" width="500">
</p>

---

### **3. Embeddings**
- Word embeddings & context
- Positional encodings
- Building embedding layers from scratch

---

### **4. Attention Mechanism**
- Self-attention (Queries, Keys, Values)
- Scaled Dot-Product Attention
- Multi-Head Attention implementation

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/Attention-mechanism.png" width="500">
</p>

---

### **5. Transformer Architecture**
- Encoder vs Decoder
- Feed Forward Networks
- Layer Normalization & Residual Connections
- Building a Transformer Block

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/15/Transformer_architecture.png" width="600">
</p>

---

### **6. Training the Model**
- Preparing datasets
- Cross-Entropy Loss
- Backpropagation basics
- Optimizers (SGD, Adam, etc.)
- Training loop implementation

---

### **7. Building a GPT-like Model**
- Stacking Transformer blocks
- Masked self-attention
- Inference process
- Sampling methods: Greedy, Top-k, Temperature scaling

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_architecture.png" width="600">
</p>

---

### **8. Scaling & Improvements**
- Training efficiency
- Model scaling laws
- Memory optimizations
- Deeper multi-layer transformer models

---

### **9. Applications of LLMs**
- Text generation
- Summarization
- Question answering
- Dialogue systems

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/ChatGPT_logo.svg" width="150">
</p>

---

### **10. Fine-tuning & Transfer Learning**
- Pretrained vs fine-tuned models
- Domain adaptation
- Low-Rank Adaptation (LoRA), Adapters
- Practical fine-tuning workflow

---

### **11. RAG & Advanced Concepts**
- Embeddings for retrieval
- Vector databases
- Retrieval Augmented Generation (RAG)
- Evaluating LLM performance

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/rag.png" width="600">
</p>

---

## 🚀 Project Goals

By following this repo, I will:
- Implement a **mini GPT** from scratch
- Train & test models on sample datasets (e.g., Shakespeare text, movie dialogues)
- Explore **RAG** for knowledge-grounded responses
- Fine-tune small LLMs on custom data
- Document learning step by step

---

## ⚡ Tech Stack

- **Python** (core implementation)  
- **PyTorch** (deep learning framework)  
- **NumPy, Pandas** (data handling)  
- **Matplotlib/Seaborn** (visualizations)  
- **Jupyter Notebooks** (experiments)

---

## 📌 Progress Tracker

- [ ] Introduction & Basics  
- [ ] Tokenization & Embeddings  
- [ ] Attention & Transformer Block  
- [ ] Training Loop  
- [ ] Mini GPT Implementation  
- [ ] Scaling Experiments  
- [ ] Applications (Text Gen, QA, Chatbot)  
- [ ] Fine-tuning & RAG  

---

## 📖 References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)  
- [HuggingFace Docs](https://huggingface.co/docs)  

---

<p align="center">🚀 Learning, Building, and Documenting my LLM journey step by step!</p>
