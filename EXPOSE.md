

## 1  Introduction  

### 1.1  Application Domain  
Since 2020 the volume of research on Large‑Language‑Models (LLMs) and related AI techniques has exploded. arXiv alone now publishes hundreds of AI‑centric papers **per week**, making it difficult for researchers, engineers and decision‑makers to keep track of what is emerging, consolidating or fading. Industry trends and terms are shaped by the most prominent research papers, and are often cited by leading AI companies and voices at conferences, gaining popularity via social media.

### 1.2  Problem Statement  
Traditional keyword searches or manual surveys no longer scale. Topic‑modeling approaches such as LDA struggle with short, jargon‑rich abstracts, while dashboards curated by humans quickly become outdated. There is a need for an **automated, reproducible pipeline** that (i) groups current AI literature into coherent topics and (ii) reveals how the prominence of these topics evolves over time.

### 1.3  Research Hypotheses  
- **H1 – Topic Detectability:** Transformer‑based sentence embeddings combined with HDBSCAN clustering (via BERTopic) can identify coherent, interpretable topic clusters in recent AI literature with a coherence score **≥ 0.40 Cᵥ**.  
- **H2 - Trend Differentiation:** At least three AI sub‑topics (e.g., *"LoRA fine‑tuning", "RLHF", "Model Context Protocol"*) show a **statistically significant** upward or downward trend (p < 0.05) between 2020 and 2025.

---

## 2  Approach & Preliminary Results  

### 2.1  Planned Experiments / Tests  
| # | Experiment | Output | Success Criterion |
|---|-----------|--------|-------------------|
| E1 | **Clustering Quality** – run BERTopic with three embedding models (MiniLM, Instructor‑XL, Berteley‑Sci) | Topic list + coherence scores | ≥ 0.40 Cᵥ for best model |
| E2 | **Temporal Topic Trends** – compute monthly topic frequencies, fit linear regression | Slope & p‑value for each topic | ≥ 3 topics with \|slope\| > 0 and p < 0.05 |
| E3 | **Manual Interpretability Check** – label 5 random papers per top‑5 topics | Confusion matrix of human labels | ≥ 80 % agreement |

### 2.2  Development Tasks  
- Python pipeline (arXiv fetch → embedding → BERTopic → trend plots)  
- Helper scripts for coherence computation & regression testing  
- Optional Streamlit dashboard for interactive exploration  

### 2.3  Open Questions  
- Optimal embedding size vs. RAM limits on local machine  
- Choice of time‑bin granularity (monthly vs. quarterly)  
- Whether Berteley's scientific stop‑word list increases coherence

### 2.4  Preliminary Signals  
View live demo of the pipeline: https://maxarnold.github.io/topic-trends/

---

## 3  Preliminary Structure of the Thesis  

1. **Abstract**  
2. **Introduction & Motivation**  
3. **Related Work** (topic modelling, BERTopic, research‑trend mining)  
4. **Methodology**  
   - Data acquisition  
   - Embedding selection  
   - BERTopic configuration  
   - Trend‑analysis metrics  
5. **Implementation** (software architecture, scripts, reproducibility)  
6. **Results**  
   - Cluster descriptions  
   - Trend graphs  
7. **Evaluation** (quantitative + qualitative)  
8. **Discussion & Limitations**  
9. **Conclusion & Future Work**  
10. **References**  
11. **Appendices** (code snippets, extended tables)

---

## 4  Roadmap (10 weeks total / 6 effective coding weeks)  

| Week | Milestone |
|------|-----------|
| 0 (01 Jul) | Exposé submitted & approved |
| 1 | Final data‑schema, fetch 5 k abstracts |
| 2 | Embedding benchmark (MiniLM vs. Instructor vs. Berteley) |
| 3 | BERTopic tuning, coherence evaluation (E1) |
| 4 | Trend extraction, regression tests (E2) |
| 5 | Manual interpretability study (E3), draft figures |
| 6 | Buffer for fixes + start full write‑up |
| 7 | Write Related Work & Methodology chapters |
| 8 | Write Results & Evaluation chapters |
| 9 | Complete Discussion, Conclusion, proofreading |
| 10 (Early Sep) | Final submission |

---

## 5  References  

1. Grootendorst, M. (2022). *BERTopic: Neural topic modelling with class-based TF-IDF.* arXiv:2203.05794. <https://arxiv.org/abs/2203.05794>
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* arXiv:1908.10084. <https://arxiv.org/abs/1908.10084>
3. Bianchi, F. et al. (2023). *BERTeley: Domain-adapted BERTopic for scientific corpora.* *Applied AI Letters.* https://doi.org/10.1016/j.dim.2024.100066
4. arXiv API documentation. <https://info.arxiv.org/help/api>  
5. Röder, M., Both, A., & Hinneburg, A. (2015). *Exploring the space of topic coherence measures.* https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
6. Devlin, J., & Chang, M.-W. (2018). *Open-sourcing BERT: State-of-the-art pre-training for NLP.* Google AI Blog. ACM DL. <https://dl.acm.org/doi/10.1145/2684822.2685324>
7. Chagnon, E., Pandolfi, R., Donatelli, J., & Ushizima, D. (2023). *Benchmarking topic models on scientific articles using BERTeley.* *Intelligent Systems with Applications*, 17, 200059. <https://www.sciencedirect.com/science/article/pii/S2949719123000419>  
8. Tillmann, A. (2025). *Literature Review of Multi-Agent Debate for Problem-Solving.* arXiv:2506.00066 [cs.CL]. <https://arxiv.org/html/2506.00066v1>  
9. Angelov, D. (2020). *Top2Vec: Distributed representations of topics.* arXiv:2008.09470. <https://arxiv.org/abs/2008.09470>
10. Hoyle, A., Goel, P., Hieronymus, C., & Boyd-Graber, J. (2021). *Is automated topic model evaluation broken? The incoherence of coherence.* *Advances in Neural Information Processing Systems*, 34. arXiv:2106.05660. <https://arxiv.org/abs/2106.05660>

---