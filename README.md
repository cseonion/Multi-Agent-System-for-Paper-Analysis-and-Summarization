# Automated Research Paper Analysis and Summarization

본 과제는 LangGraph를 기반으로 이뤄졌습니다.
제안된 모든 테스트 케이스에 대한 결과가 포함되어 있습니다.

## Repo

```
main.py     				# run
run.ipynb   				# run(recommend)
📂[agents]					# LangGraph agents
	analysis_comparison_agent.py
	analysis_cross_domain_agent.py
	analysis_lit_review_agent.py
	analysis_plan_router.py
	domain_agent.py
	summary_agent.py
	write_agent.py
	📂[tools] 					# tools for agents
		arxiv.py      				# arXiv searching tool
		python_repl.py  			# pythonREPL tool
		vectorstore.py  			# vectorstore retriever tool
		web_search.py   			# web search tool
📂[cache]   				# results after execution
	(...)
📂[config]
	agent_config.json   		# agents llm config
	agent_llm.py 				# agents llm calling
	logging_config.py   		# logger config
📂[src]
	graph.py      				# graph(LangGraph)
	state.py      				# state(LangGraph)
	load_doc.py   				# document(pdf) loader
	tracking.py   				# input/output token tracking
📂[test]       	  
	📂[case1]     				# paper for "single-paper analysis"
	📂[case2]     				# papers for "multi-paper comparison"
	📂[case3]     				# papers for "literature review synthesis"
	📂[caseE]     				# paper for "cross-domain paper"
📂[test_output] 			# example outputs
	(...)
```

## Topology

```
[PDF Ingest]
   → (Extract)
      └→ (Documant Parsing) → (Paragraph Chunking) → (Embedding)
   → [Summary Subgraph]
	  └→ (Summary: section) → (Summary: paper)
   → [Domain Identity Agent]
   → [Analysis Plan Router] ─┬─→ [Comparison Agent]
                          	 ├─→ [LitReview Agent]
							 └─→ [Cross-Domain Agent]
   → [Writing Agent]
```

![graph workflow](./image/graph.png)

## Agent descriptions

자세한 내용은 `agents/Info.md`를 참고하세요.

## Tools
- Vectorstore Retriever
	- 분석 에이전트 모두에게 제공
  	- 벡터스토어에 저장된 논문 원문에 접근하기 위함
- Web Searching
	- DuckDuckGo API
   	- 분석 에이전트 모두에게 제공
	- 부족한 정보 보충을 위함
- ArXiv Searching
	- LangChain 기본 제공 API
 	- 분석 에이전트 중 Literature Review Agent에 제공
 	- 분석 방향 제시에 활용됨
- Python Coding
	- LangChain 기본 제공 API(pythonREPL)
 	- Writing Agent에 제공
  	- 파이썬으로 구현 가능한 요소가 있을 경우, 간단한 의사 코드 작성을 위함
---

# Installation

## 1. GROBID 설치

**Docker는 필수적으로 설치 되어 있어야 합니다.** 설치되어 있지 않다면, 다음을 참고하세요. [링크](https://docs.docker.com/get-started/docker-overview/)
GROBID는 scientific paper parsing에 특화된 ML 라이브러리로, 본 시스템에서는 입력된 pdf에서 텍스트를 추출하는 tool로 사용됩니다.
    - GROBID 설치에 대한 세부 정보는 다음을 참고하세요. [링크](https://grobid.readthedocs.io/en/latest/Grobid-docker/)
- 아래 두 옵션 중 하나를 선택하여 터미널에서 설치 및 구동하여 docker container가 실행되어야 합니다.

### ❌ opt1. **Deep Learning and CRF image**

- 딥러닝 모델을 기반으로 가장 강력한 추출 성능을 내는 버전입니다.
- 설치에 10GB의 공간이 필요합니다.
- 단, 현재(25.08.16 기준) Linux OS에서만 지원됩니다.
    - 본 과제에서 사용되지 않았습니다.

```
docker pull grobid/grobid:0.8.2
```

```
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2
```

### ✅ opt2. **CRF-only image**

- CRF 모델을 기반으로 추출을 수행하는 lightweight 버전입니다.
- 설치에 300MB의 공간이 필요합니다.
- Linux/Windows/Mac OS에서 작동됩니다.
    - 본 과제는 해당 image를 기반으로 수행되었습니다.

```
docker pull lfoppiano/grobid:0.8.2
```

```
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.2
```

## 2. 가상환경 세팅

- 가상환경 설정
    
    ```
    conda create -n taskpaper python=3.11 
    conda activate taskpaper
    ```
    
- 패키지 설치
    
    ```
    pip install -r requirments.txt
    ```
    

---

## Execution

- `run.ipynb`를 확인하세요.

## Results

실행 및 분석이 완료되면 `cache` 폴더에 결과가 저장됩니다.
```
📁[cache]
	📁[(timestamp)]
		📁[paper_1]
			📁[vectorstore]		# raw text vector stored in local
			📁[summaries]		# all summary included
				(...)
		📁[paper_2]
			(...)
		📁(...)
		eval.json				# evaluation results
		process.log				# log
		execution_tracking.json # tracking results
		report.md				# final output of system
```
# Tests

## Paper list

각 테스트 케이스를 확인하기 위해 선정된 논문들은 다음과 같습니다.

### case1: A recent machine learning research paper
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), 2025

### case2: 3 papers on similar ML topics from different years
- [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), 2017
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/?utm_campaign=The+Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_m9bbH_7ECE1h3lZ3D61TYg52rKpifVNjL4fvJ85uqggrXsWDBTB7YooFLJeNXHWqhvOyC), 2019
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692), 2019
  
### case3: 5 papers from related but distinct subfields
'음악'이라는 공통된 주제로 논문을 선정하였습니다.

- [Alarm Tones, Voice Warnings, and Musical Treatments: A Systematic Review of Auditory Countermeasures for Sleep Inertia in Abrupt and Casual Awakenings](https://www.mdpi.com/2624-5175/2/4/31?ref=nightwater.email), 2020
- [Music, memory and emotion](https://link.springer.com/article/10.1186/jbiol82), 2008
- [In-Context Prompt Editing for Conditional Audio Generation](https://arxiv.org/abs/2311.00895), 2023
- [IteraTTA: An interface for exploring both text prompts and audio priors in generating music with text-to-audio models](https://arxiv.org/abs/2307.13005), 2023
- [MusicEval: A Generative Music Dataset with Expert Ratings for Automatic Text-to-Music Evaluation](https://ieeexplore.ieee.org/abstract/document/10890307?casa_token=mrdNmZeDQw8AAAAA:UGSNihuCNj9VcuPwp0YYuynz86jnQHpglNc2mAzZJciiy7DQyxkxMJFeiarecN-B0ZGoH7vyFw), 2025
  
### caseE: Interdisciplinary paper combining ML with biology
- [Diffusion on PCA-UMAP Manifold: The Impact of Data Structure Preservation to Denoise High-Dimensional Single-Cell RNA Sequencing Data](https://www.mdpi.com/2079-7737/13/7/512), 2024

## Test results

`test_output` 폴더 내 모든 테스트의 결과가 존재합니다.
시스템이 최종적으로 출력한 모든 정보를 담고 있습니다.

# Presentation

https://www.notion.so/cseonion/PT-24f6ec68c109803a84a1c26d5fe3ba4f?source=copy_link
