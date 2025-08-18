# Summary Agent
- 역할
	1. 현재 섹션 텍스트와 일관성을 유지하기 위한 방법으로 이전 섹션 요약본까지 추가로 입력으로 받고 섹션 단위 요악을 수행
	2. 섹션 단위 요약본들을 기반으로 전체 요약본을 생성
입력된 논문의 수 만큼 subgraph가 동적으로 생성되며, 요약의 과정이 긴 편이기 때문에, 모든 subgraph는 send API를 통해 병렬적으로 요약을 수행한다. subgraph의 구조는 아래와 같다.
	![subgraph for summary agent](../image/subgraph.png)

# Domain Identity Agent

- 역할
	- 입력된 논문의 메인/서브 분야를 파악
	 	- main field (ex. Computer Science, Physics, Psychology)
		- sub field (ex. Machine Learning, Quantum Physics, Cognitive Psychology)
Analysis Plan Router의 기준이 되는 출력값을 반환한다.

# Analysis Plan Router

- 역할

# Cross-domain Agent

- Cross-domain 논문 분석
	- instruction
		```
  		```
	- prompt
   		```
     	```
 		- `~~~`: 
	- output
 		- ㅇㅇ

# Comparison Agent

- 다중 논문 비교
	- instruction
		```
  		```
	- prompt
   		```
     	```
 		- input
 		- `~~~`: 
	- output
 		- ㅇㅇ

# Literature Review Agent

- 문헌 리뷰
	- instruction
		```
  		```
	- prompt
   		```
     	```
 		- `~~~`: 
	- output
 		- ㅇㅇ
 
# Writing Agent
