# Automated Research Paper Analysis and Summarization

## Repo

```
[Task2]
   
   main.py     # 실행
   run.ipynb   # 실행(권장)
   → [agents] → [tools] ─┬─→ arxiv.py        # arXiv 서치 툴
			|			 ├─→ python_repl.py  # pythonREPL 툴
        	|            ├─→ vectorstore.py  # vectorstore retriever 툴
		    |            └─→ web_search.py   # 웹 서치 툴
		    |                
	        ├─→ analysis_comparison_agent.py
		    ├─→ analysis_cross_domain_agent.py
		    ├─→ analysis_lit_review_agent.py
		    ├─→ analysis_plan_router.py
		    ├─→ domain_agent.py
		    ├─→ summary_agent.py
		    └─→ write_agent.py
   → [cache]   # 실행 결과 저장
   → [config]
			└─→ logging_config.py   # 로깅 config
   → [src]
			├─→ graph.py      # 그래프 정의
  		  	├─→ load_doc.py   # document(pdf) loader
	      	├─→ state.py      # 상태 정의
	      	└─→ tracking.py   # 출력 비용 추적 (fail)
   → [test]    # test case 논문들
			├─→ (...)
		  (...)'
   → [test_output] # test case 출력물
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

### Summary Agent

1. 섹션 단위 요약본 생성
    
    
    | Input | - 한 섹션에 해당하는 텍스트 전체
    - 이전 섹션 요약본 |
    | --- | --- |
    | Output | 섹션 요약본 |
2. 전체 요약본 생성
    
    
    | Input | 섹션 단위 요약본을 병합한 문서 |
    | --- | --- |
    | Output | 논문 전체 요약본 |

### Domain Identity Agent

- 논문 도메인 판별
    
    
    | Input | 논문 전체 요약본 |
    | --- | --- |
    | Output | - 주요 분야
    - 하위 분야 |

### Analysis Plan Router

- 분석 agent 라우팅
    
    
    | Input | - 주요 분야
    - 하위 분야 |
    | --- | --- |
    | Output | 분석법 |

### Cross-domain Agent

- Cross-domain 논문 분석
    
    

### Comparison Agent

- 다중 논문 비교

### Literature Review Agent

- 문헌 리뷰

### Ideation Agent

- 연구적 아이디어 제공

---

# Installation

## 1. GROBID 설치

- **Docker는 필수적으로 설치 되어 있어야 합니다.**
    - 설치되어 있지 않다면, 다음을 참고하세요. [링크](https://docs.docker.com/get-started/docker-overview/)
- GROBID는 scientific paper parsing에 특화된 ML 라이브러리로, 본 시스템에서는 입력된 pdf에서 텍스트를 추출하는 tool로 사용됩니다.
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

# Execution

- `run.ipynb`를 확인하세요.

# Results

- 실행 및 분석이 완료되면 `cache` 폴더에 결과가 저장됩니다.
	- 📁 {datetime}
 		- 📁 {paper_1}
   			- 📁 {vector store}
      			- faiss
         		- pkl
      		- 📁 {summary results}
        		- (...)
	    - 📁 {paper_2}
     		- (...)
   		- log
     	- traking
      	- final_report.md 	
