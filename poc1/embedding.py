import fitz  # PyMuPDF
import faiss
import numpy as np
import sys
import os
import pickle
import json

MODEL_PATH = os.path.abspath("../")  # 예: 한 단계 바깥 폴더
sys.path.append(MODEL_PATH)
from model import initialize_llm, initialize_embedding

# LangChain 관련 임포트 추가
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

# Azure OpenAI 임베딩 설정
embeddings = initialize_embedding("text-embedding-3-large")

# 저장 경로 설정
FAISS_INDEX_PATH = "./faiss_/faiss_index.bin"
CHUNKS_PATH = "./faiss_/chunks.pkl"
METADATA_PATH = "./faiss_/metadata.json"


# PDF 로딩 및 페이지별 텍스트 추출
def extract_pdf_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    metadata = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            chunks.append(text)
            metadata.append({"page": i + 1})
    return chunks, metadata


# 텍스트를 Azure OpenAI 임베딩 모델로 임베딩
def embed_texts(texts):
    try:
        # LangChain의 embed_documents 사용 (배치 처리 자동 지원)
        embeddings_list = embeddings.embed_documents(texts)
        return np.array(embeddings_list).astype("float32")
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        # 오류 발생 시 개별 텍스트로 재시도
        embeddings_list = []
        for i, text in enumerate(texts):
            try:
                embedding = embeddings.embed_query(text)
                embeddings_list.append(embedding)
                if (i + 1) % 10 == 0:
                    print(f"진행 상황: {i + 1}/{len(texts)} 완료")
            except Exception as e2:
                print(f"개별 텍스트 임베딩 실패 (인덱스 {i}): {e2}")
                # 실패한 경우 0 벡터로 대체
                embeddings_list.append([0.0] * 1536)  # Ada-002는 1536차원

        return np.array(embeddings_list).astype("float32")


# FAISS 인덱스 구축
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# 검색 함수 추가
def search_similar_chunks(query, index, chunks, metadata, top_k=5):
    """쿼리와 유사한 청크 검색"""
    try:
        # LangChain의 embed_query 사용
        query_embedding = embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype("float32")

        # FAISS 검색
        scores, indices = index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks):
                results.append(
                    {
                        "text": chunks[idx],
                        "metadata": metadata[idx],
                        "score": float(score),
                    }
                )

        return results
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return []


# FAISS 인덱스와 데이터 저장
def save_faiss_data(
    index,
    chunks,
    metadata,
    index_path=FAISS_INDEX_PATH,
    chunks_path=CHUNKS_PATH,
    metadata_path=METADATA_PATH,
):
    """FAISS 인덱스와 메타데이터를 디스크에 저장"""
    print(f"FAISS 인덱스를 {index_path}에 저장 중...")
    faiss.write_index(index, index_path)

    print(f"청크 데이터를 {chunks_path}에 저장 중...")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"메타데이터를 {metadata_path}에 저장 중...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("모든 데이터 저장 완료!")


# FAISS 인덱스와 데이터 불러오기 (원본)
def load_faiss_data_raw(
    index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH, metadata_path=METADATA_PATH
):
    """저장된 FAISS 인덱스와 메타데이터를 불러오기 (원본 형태)"""
    try:
        print(f"FAISS 인덱스를 {index_path}에서 불러오는 중...")
        index = faiss.read_index(index_path)

        print(f"청크 데이터를 {chunks_path}에서 불러오는 중...")
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        print(f"메타데이터를 {metadata_path}에서 불러오는 중...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("모든 데이터 불러오기 완료!")
        return index, chunks, metadata

    except FileNotFoundError as e:
        print(f"저장된 파일을 찾을 수 없습니다: {e}")
        return None, None, None
    except Exception as e:
        print(f"데이터 불러오기 중 오류 발생: {e}")
        return None, None, None


# FAISS 인덱스와 데이터 불러오기 (LangChain VectorStore 형태)
def load_faiss_data(
    index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH, metadata_path=METADATA_PATH
):
    """저장된 FAISS 인덱스와 메타데이터를 LangChain VectorStore로 불러오기"""
    try:
        # 원본 데이터 로드
        index, chunks, metadata = load_faiss_data_raw(
            index_path, chunks_path, metadata_path
        )

        if index is None:
            return None, None, None

        print("LangChain VectorStore로 변환 중...")

        # 청크를 Document 객체로 변환
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata=metadata[i] if i < len(metadata) else {"page": i + 1},
            )
            documents.append(doc)

        # InMemoryDocstore 생성
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        # 문서를 docstore에 추가
        for i, doc in enumerate(documents):
            doc_id = str(i)
            docstore.add({doc_id: doc})
            index_to_docstore_id[i] = doc_id

        # LangChain FAISS VectorStore 생성
        vectorstore = LangChainFAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        print("LangChain VectorStore 변환 완료!")
        return vectorstore, chunks, metadata

    except Exception as e:
        print(f"VectorStore 변환 중 오류 발생: {e}")
        return None, None, None


# 저장된 데이터 존재 여부 확인
def check_saved_data_exists(
    index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH, metadata_path=METADATA_PATH
):
    """저장된 FAISS 데이터가 존재하는지 확인"""
    return (
        os.path.exists(index_path)
        and os.path.exists(chunks_path)
        and os.path.exists(metadata_path)
    )


# 전체 파이프라인 수정
def process_pdf_to_faiss(pdf_path, force_rebuild=False):
    """PDF를 처리하여 FAISS 인덱스 생성 (캐시 기능 포함)"""

    # 저장된 데이터가 있고 강제 재구축이 아닌 경우 불러오기
    if not force_rebuild and check_saved_data_exists():
        print("저장된 FAISS 인덱스를 발견했습니다. 불러오는 중...")
        vectorstore, chunks, metadata = load_faiss_data()
        if vectorstore is not None:
            # VectorStore에서 원본 인덱스 추출
            index = vectorstore.index
            print(f"불러온 인덱스 정보: 벡터 수 {index.ntotal}")
            return vectorstore, chunks, metadata
        else:
            print("저장된 데이터 불러오기 실패. 새로 생성합니다.")

    # 새로 생성하는 경우
    print("PDF에서 텍스트 추출 중...")
    chunks, metadata = extract_pdf_chunks(pdf_path)
    print(f"총 {len(chunks)}개 청크 추출됨")

    print("텍스트 임베딩 생성 중...")
    embeddings_array = embed_texts(chunks)
    print(f"임베딩 생성 완료: {embeddings_array.shape}")

    print("FAISS 인덱스 구축 중...")
    index = create_faiss_index(embeddings_array)
    print("인덱스 구축 완료")

    # 생성된 데이터 저장
    save_faiss_data(index, chunks, metadata)

    # 저장 후 VectorStore로 변환해서 반환
    vectorstore, chunks, metadata = load_faiss_data()
    return vectorstore, chunks, metadata


# 사용 예시
def run(pdf_path="../data/data1_manual.pdf"):
    # pdf_path = "../data/data1_manual.pdf"  # 여기에 PDF 파일 경로 입력

    try:
        # force_rebuild=True로 설정하면 강제로 다시 생성
        vectorstore, chunks, metadata = process_pdf_to_faiss(
            pdf_path, force_rebuild=False
        )

        # VectorStore에서 원본 인덱스 추출
        index = vectorstore.index
        print("FAISS 인덱스에 저장된 벡터 수:", index.ntotal)
        print("첫 페이지 청크 예시:", chunks[0][:200])

        # 검색 테스트
        test_query = "도시가스 요금"  # 더 구체적인 검색어로 변경
        print(f"\n검색 쿼리: {test_query}")
        results = search_similar_chunks(test_query, index, chunks, metadata)

        print(f"\n상위 {len(results)}개 결과:")
        for i, result in enumerate(results):
            print(
                f"{i+1}. 페이지 {result['metadata']['page']}, 점수: {result['score']:.4f}"
            )
            print(f"   텍스트: {result['text'][:100]}...")
            print()

    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        print("API 키, 엔드포인트, 모델 이름을 확인해주세요.")


# run()
