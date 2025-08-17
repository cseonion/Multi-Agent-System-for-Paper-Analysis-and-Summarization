from langchain_community.document_loaders.parsers import GrobidParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from copy import deepcopy
from datetime import datetime
from src.tracking import track_agent
import os
import logging
from datetime import datetime

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# 현재 모듈 로거 생성 (main.py에서 설정한 로깅 사용)
logger = logging.getLogger(__name__)

def document_loader(doc_path, segment_sentences: bool = False):
    """
    Load documents from a given path using GrobidParser.

    Args:
        doc_path (str): Path to the document or directory containing documents.
        segment_sentences (bool): Whether to segment sentences in the document.
    Returns:
        list: List of loaded documents.
    """
    if len(doc_path) == 1 and doc_path[0].endswith('.pdf'):
        loader = GenericLoader.from_filesystem(
            path=doc_path[0],
            parser=GrobidParser(segment_sentences=segment_sentences),
            show_progress=True
        )
    elif len(doc_path) == 1 and doc_path[0].endswith('/'):
        loader = GenericLoader.from_filesystem(
            path=doc_path[0],
            glob="*",
            suffixes=[".pdf"],
            parser=GrobidParser(segment_sentences=segment_sentences),
            show_progress=True
        )
    docs = loader.load()
    return docs

def docs_into_tuples(docs):
    """
    Extract metadata from documents and return a list of tuples.
    : title of the paper, section number, section title, and document object.
    
    Args:
        docs (list): List of documents to extract metadata from.

    Returns:
        list: List of tuples containing extracted metadata.
    """
    meta_tuples = []
    for doc in docs:
        meta = doc.metadata
        paper_title = meta.get('paper_title')
        section_number = meta.get('section_number')
        section_title = meta.get('section_title')
        meta_tuples.append((paper_title, section_number, section_title, doc))
    return meta_tuples

def split_by_title(meta_tuples):
    """
    Split metadata tuples by paper title. (Only for 2+ papers)
    
    Args:
        meta_tuples (list): List of tuples containing metadata.

    Returns:
        dict: Dictionary where keys are paper titles and values are lists of tuples
              containing section number, section title, and document object.
    """
    title_dict = {}
    for paper_title, section_number, section_title, doc in meta_tuples:
        if paper_title not in title_dict:
            title_dict[paper_title] = []
        title_dict[paper_title].append((section_number, section_title, doc))
    return title_dict

def postprocess_section_numbers(title_dict):
    # section_number가 None인 경우 후처리
    # 마지막 섹션 정보로 대체하거나 'etc'로 대체
    title_dict_post = {}
    for paper_title, items in title_dict.items():
        # deepcopy로 원본 보존
        new_items = deepcopy(items)
        # 마지막 section_number가 있는 데이터의 인덱스 찾기
        last_sec_idx = -1
        last_sec_num = None
        last_sec_title = None
        for idx, (sec_num, sec_title, doc) in enumerate(new_items):
            if sec_num is not None and sec_num != 'None':
                last_sec_idx = idx
                last_sec_num = sec_num
                last_sec_title = sec_title
        # 후처리
        prev_sec_num = None
        prev_sec_title = None
        for idx, (sec_num, sec_title, doc) in enumerate(new_items):
            if sec_num is not None and sec_num != 'None':
                prev_sec_num = sec_num
                prev_sec_title = sec_title
            else:
                if idx <= last_sec_idx:
                    # 본문 내: 직전 섹션 정보로 대체
                    new_items[idx] = (prev_sec_num, prev_sec_title, doc)
                else:
                    # 마지막 이후: etc로 대체
                    new_items[idx] = ('etc', sec_title, doc)
        title_dict_post[paper_title] = new_items
    
    return title_dict_post

def merge_sections(title_dict_post):
    """
    Merge sections by combining paragraphs within each section.

    Args:
        title_dict_post (dict): Dictionary where keys are paper titles and values are lists of tuples
                                containing section number, section title, and document object.

    Returns:
        dict: Dictionary where keys are paper titles and values are dictionaries with section numbers as keys
              and combined paragraphs as values.
    """
    # section 단위로 문단을 합쳐 새로운 dictionary 생성
    section_docs_dict = {}
    for paper_title, items in title_dict_post.items():
        section_dict = {}
        for sec_num, sec_title, doc in items:
            key = (sec_num, sec_title)
            if key not in section_dict:
                section_dict[key] = []
            # 문단 텍스트 추가
            section_dict[key].append(doc.page_content if hasattr(doc, 'page_content') else doc.metadata.get('text', ''))
        # 섹션별로 텍스트 합치기
        merged_section = {}
        for (sec_num, sec_title), paras in section_dict.items():
            merged_section[(sec_num, sec_title)] = '\n'.join([p for p in paras if p])
        section_docs_dict[paper_title] = merged_section
    return section_docs_dict

def get_section_list(section_docs_dict):
    """
    Extract section titles from the merged sections.

    Args:
        section_docs_dict (dict): Dictionary where keys are paper titles and values are dictionaries with section numbers as keys
                                  and combined paragraphs as values.

    Returns:
        dict: Dictionary where keys are paper titles and values are lists of section titles.
    """
    section_titles = {}
    for paper_title, sec_dict in section_docs_dict.items():
        titles = [(sec_num, sec_title) for (sec_num, sec_title) in sec_dict.keys()]
        section_titles[paper_title] = titles
    return section_titles

def build_section_documents(section_docs_dict):
    """
    Build a dictionary of section documents from the merged sections.

    Args:
        section_docs_dict (dict): Dictionary where keys are paper titles and values are dictionaries with section numbers as keys
                                  and combined paragraphs as values.

    Returns:
        dict: Dictionary where keys are paper titles and values are lists of Document objects for each section.
    """
    section_documents = {}
    for paper_title, sec_dict in section_docs_dict.items():
        doc_list = []
        for (sec_num, sec_title), text in sec_dict.items():
            meta = {
                'paper_title': paper_title,
                'section_number': sec_num,
                'section_title': sec_title
            }
            doc_list.append(Document(page_content=text, metadata=meta))
        section_documents[paper_title] = doc_list
    return section_documents

def make_vectorstore(state, section_documents):
    """
    Create a vector store from the loaded documents.

    Args:
        section_documents (dict): Dictionary where keys are paper titles and values are lists of Document objects for each section.

    Returns:
        tuple: (vectorstores, vectorstores_path, cache_dir)
    """
    logger.info("🔧 Starting vectorstore creation")
    
    vectorstores = {}
    vectorstores_path = {}

    if not section_documents or len(section_documents) == 0:
        raise ValueError("No section_documents provided for vectorstore creation")

    try:
        embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'mps'}  # Force CPU to avoid tensor device issues
        )
        logger.info(f"✅ Embedder initialized with model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        raise RuntimeError(f"Failed to initialize embedder: {e}")

    cache_dir = state["cache_dir"]

    for paper_title, docs in section_documents.items():
        logger.info(f"🔄 Creating vectorstore for: {paper_title}")
        logger.debug(f"   📄 Processing {len(docs)} sections")
        
        # vectorstore 생성
        vectorstore = FAISS.from_documents(docs, embedder)
        vectorstores[paper_title] = vectorstore

        # 저장 경로 지정
        save_path = os.path.join(cache_dir+paper_title, "vectorstore/")
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        vectorstores_path[paper_title] = save_path
        logger.info(f"💾 Vectorstore for '{paper_title}' saved to {save_path}")

    logger.info(f"✅ All vectorstores created successfully ({len(vectorstores)} papers)")
    return vectorstores, vectorstores_path

@track_agent("extract")
def extract(state):
    # 최종적으로 load_doc.py에서 정의한 함수를 합쳐 state에 필요한 정보 전달 및 벡터스토어 저장
    logger.info("🔄 Document extraction process started")
    logger.info(f"📂 Processing paths: {state['path']}")
    
    docs = document_loader(state['path'], segment_sentences=False)
    logger.info(f"📄 Loaded {len(docs)} documents")
    
    meta_tuples = docs_into_tuples(docs)
    title_dict = split_by_title(meta_tuples)
    title_dict_post = postprocess_section_numbers(title_dict)
    section_docs_dict = merge_sections(title_dict_post)
    section_list = get_section_list(section_docs_dict)
    section_documents = build_section_documents(section_docs_dict)
    
    logger.info(f"📊 Found papers: {list(section_documents.keys())}")

    # state에 벡터스토어 정보 추가
    vectorstores, vectorstores_path = make_vectorstore(state, section_documents)
    
    logger.info("✅ Document extraction completed successfully")
    
    return {
        "paper_title": list(section_documents.keys()),  # 논문 제목 리스트
        "paper_sections": section_list,
        "vectorstores": vectorstores,
        "vectorstores_path": vectorstores_path,
    }