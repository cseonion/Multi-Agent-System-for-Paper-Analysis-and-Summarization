from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re
import json

from rouge_score import rouge_scorer  
from bert_score import score as bertscore_score 
import textstat  

# ---- LLM-as-a-judge 설정 영역 ----
# 필요 시 이 영역의 텍스트를 수정해 에이전트별 심사 기준을 커스터마이즈하세요.
JUDGE_AGENT_CONFIG = {
    "agent_name": "eval_judge",  # config/agent_config.json 에서 이 이름의 LLM 설정을 사용
    "response_schema": {
        "factual_consistency": "원문 근거 대비 사실 일치도. min=1, max=5",
        "support_coverage": "핵심 주장/요소가 근거로 지원되는 비율. min=1, max=5",
        "hallucination_detected": "true/false (환각·허구 정보 포함 여부)",
        "issues": "문제 목록 (환각, 수치 오류, 과장 등)",
        "rationale": "감점에 대한 간단한 근거 설명 (원문 근거 인용 포함)",
    },
}

# 에이전트별 기본 심사 기준 템플릿
JUDGE_CRITERIA_TEMPLATES: Dict[str, str] = {
    "domain_agent": (
        "- 원문이 예측한 main/sub-field에 해당 되는지\n"
        "- 과도한 일반화/환각 여부 판단"
    ),
    "analysis_plan_router": (
        "- 입력 조건(논문 수, 도메인 유사/상이 여부)에 비춰 선택된 플랜의 타당성\n"
        "- 대안 플랜이 더 합리적이면 사유 명시"
    ),
    "summary_agent(section)": (
        "- 현재 섹션 요약이 해당 섹션 원문과 사실 일치하는지\n"
    ),
    "summary_agent(final)": (
        "- 최종 요약이 섹션 요약들의 핵심을 정확히 통합/커버하는지\n"
        "- 과장/환각 없이 사실 기반으로 서술했는지"
    ),
    "comparison_agent": (
        "- 각 논문에 대한 주장(모델/데이터/강점/약점)이 증거로 지지되는지\n"
        "- 비교 결론의 정합성 및 근거 인용"
    ),
    "cross_domain_agent": (
        "- 도메인 간 연결·함의가 증거에 기반하는지\n"
        "- 과도한 일반화/환각 지적"
    ),
    "lit_review_agent": (
        "- 시간축/연구 흐름/한계·공백·제안이 증거와 부합하는지\n"
        "- 연도/저자/핵심 결과의 근거 확인"
    ),
    "write_agent": (
        "- 기사/블로그 내용이 분석 결과/증거와 모순 없는지\n"
        "- 과장/환각/부정확한 수치 지적"
    ),
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict academic evaluator. Your job is to assess the candidate text against the provided evidences for factual consistency and hallucinations.\n"
    "Follow the criteria for the specific task, and ALWAYS return valid JSON only with the following keys: \n"
    "{keys}\n"
)

JUDGE_USER_PROMPT_TEMPLATE = (
    "[Task] {task}\n\n"
    "[Criteria]\n{criteria}\n\n"
    "[Candidate]\n{candidate}\n\n"
    "[Evidences]\n{evidences}\n\n"
    "Return ONLY JSON with keys: factual_consistency (0.0-1.0), support_coverage (0.0-1.0), hallucination_detected (bool), issues (list of strings), rationale (string)."
)


# ---------- 공통 유틸 ----------

def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _jaccard(a: str, b: str) -> float:
    sa = set(_normalize_text(a).split())
    sb = set(_normalize_text(b).split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _contains_headings(md: str, min_count: int = 2) -> bool:
    return len(re.findall(r"^#{1,6} ", md or "", flags=re.MULTILINE)) >= min_count


def _has_markdown_table(md: str) -> bool:
    return bool(re.search(r"\n\|.*\|\n\|[-: ]+\|", md or ""))


def rouge_l(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return float(scores["rougeL"].fmeasure)


def bertscore_f1_single(candidate: str, reference: str) -> float:
    P, R, F1 = bertscore_score([candidate], [reference], lang="en", rescale_with_baseline=True)  # type: ignore
    return float(F1.mean().item())


@dataclass
class EvalResult:
    target: str
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: float):
        self.scores[name] = float(value)

    def merge(self, other: "EvalResult", prefix: Optional[str] = None):
        p = (prefix + ":") if prefix else ""
        for k, v in other.scores.items():
            self.scores[p + k] = v
        self.details.update(other.details)


# ---------- LLM 호출 ----------

def get_judge_llm():
    """config/agent_config.json 의 eval_judge 설정을 사용해 LLM을 생성."""
    from config.agent_llm import get_llm

    return get_llm(JUDGE_AGENT_CONFIG.get("agent_name", "eval_judge"))


def _format_evidences(evidences: List[str], max_chars: int = 8000) -> str:
    joined = "\n\n".join([f"[E{idx+1}] {e}" for idx, e in enumerate(evidences or [])])
    return joined[:max_chars]


def llm_as_judge(agent: str, candidate: str, evidences: List[str], criteria_override: Optional[str] = None) -> EvalResult:
    """Judge LLM으로 사실성/커버리지/환각 여부를 평가하여 공통 스키마로 반환."""
    res = EvalResult(target=f"{agent}_judge")

    keys_desc = ", ".join([f"{k}: {v}" for k, v in JUDGE_AGENT_CONFIG["response_schema"].items()])
    system_msg = JUDGE_SYSTEM_PROMPT.format(keys=keys_desc)
    criteria = criteria_override or JUDGE_CRITERIA_TEMPLATES.get(
        agent, "- Assess factual consistency with cited evidences. - Flag hallucinations and numeric inconsistencies."
    )
    user_msg = JUDGE_USER_PROMPT_TEMPLATE.format(
        task=agent,
        criteria=criteria,
        candidate=candidate or "",
        evidences=_format_evidences(evidences or []),
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore

        llm = get_judge_llm()
        response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
        content = getattr(response, "content", str(response))
        data = _extract_json_obj(content) or {}
    except Exception as e:
        data = {}
        res.details["error"] = f"judge_llm_error: {e}"

    factual = float(data.get("factual_consistency", 0.0) or 0.0)
    coverage = float(data.get("support_coverage", 0.0) or 0.0)
    halluc = bool(data.get("hallucination_detected", False))

    res.add("factual_consistency", factual)
    res.add("support_coverage", coverage)
    res.add("hallucination_detected", 1.0 if halluc else 0.0)

    if "issues" in data:
        res.details["issues"] = data.get("issues")
    if "rationale" in data:
        res.details["rationale"] = data.get("rationale")

    res.details.setdefault("_meta", {})
    res.details["_meta"].update({"agent": agent, "evidences_count": len(evidences or [])})
    return res


# ---------- 통합 평가 함수 (에이전트 공용) ----------

def evaluate_with_judge(
    agent: str,
    candidate: str,
    evidences: List[str],
    *,
    metrics: Optional[List[str]] = None,  # ["rouge", "bertscore", "structure", "readability"] 등 선택
    references: Optional[List[str]] = None,  # rouge/bertscore 비교용 참조 텍스트 목록
    criteria_override: Optional[str] = None,
) -> EvalResult:
    """단일 함수로 모든 agent 결과를 평가.

    - 필수: agent, candidate, evidences
    - 선택: metrics (추가 지표), references (ROUGE/BERTScore 참조)
    - criteria_override: 에이전트별 기본 기준을 덮어쓰기
    """
    metrics = metrics or []

    # 1) Judge
    result = llm_as_judge(agent=agent, candidate=candidate, evidences=evidences, criteria_override=criteria_override)

    # 2) 선택적 메트릭
    # 2-1) ROUGE-L: candidate vs best reference (또는 evidences 중 최고 유사)
    if "rouge" in [m.lower() for m in metrics]:
        best_ref = None
        pool = references if references else evidences
        if pool:
            best_ref = max(pool, key=lambda r: _jaccard(candidate, r))
        if best_ref:
            result.add("rouge_l", rouge_l(candidate, best_ref))

    # 2-2) BERTScore-F1: candidate vs best reference
    if "bertscore" in [m.lower() for m in metrics]:
        best_ref = None
        pool = references if references else evidences
        if pool:
            best_ref = max(pool, key=lambda r: _jaccard(candidate, r))
        if best_ref:
            result.add("bertscore_f1", bertscore_f1_single(candidate, best_ref))

    # 2-3) 구조 신호: 헤딩/테이블 유무
    if "structure" in [m.lower() for m in metrics]:
        result.add("has_headings", 1.0 if _contains_headings(candidate) else 0.0)
        result.add("has_table", 1.0 if _has_markdown_table(candidate) else 0.0)

    # 2-4) 가독성(Flesch Reading Ease)
    if "readability" in [m.lower() for m in metrics]:
        try:
            score = float(textstat.flesch_reading_ease(candidate))
        except Exception:
            score = 0.0
        result.add("readability", score)

    return result


# 참고: LangGraph 문서의 LLM-as-a-judge 패턴을 따르되 LangSmith 의존성 없이 구성했습니다.
# 참고: agentevals 레포 아이디어(trajectory 등)를 기준 디자인에 반영할 수 있으나, 본 구현은 self-contained 평가에 집중합니다.
