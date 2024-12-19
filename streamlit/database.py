import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings  # 변경된 임포트
from langchain_chroma import Chroma  # 변경된 임포트
import torch  # GPU 정보 확인을 위해 추가

def load_database(persist_directory, collection_name, embeddings):
    """
    데이터베이스를 로드하고, GPU 정보와 문서 개수를 사이드바에 출력.
    - persist_directory: Chroma 데이터베이스의 경로
    - embedding_model_name: 임베딩 모델 이름 (기본값: intfloat/multilingual-e5-large)
    - collection_name: Chroma 데이터베이스의 컬렉션 이름
    """
    try:
        # 임베딩 모델 로드
        embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
        
        # Chroma 데이터베이스 로드
        database = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        retriever = database.as_retriever(search_kwargs={"k": 5})
        
        # GPU 정보 가져오기
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "사용 가능한 GPU 없음"
        
        # 문서 개수 확인
        doc_count = database._collection.count() if database._collection else 0
        
        # 사이드바에 정보 출력
        st.sidebar.header("시스템 정보")
        st.sidebar.write(f"**사용 중인 GPU:** {gpu_name}")
        st.sidebar.write(f"**로드된 문서 개수:** {doc_count}")
        
        # 성공 메시지
        st.success(f"✅ Chroma 데이터베이스가 성공적으로 로드되었습니다! 경로: {persist_directory}, 컬렉션: {collection_name}")
        return retriever
        
    except Exception as e:
        # 오류 메시지 출력
        st.sidebar.header("시스템 정보")
        st.sidebar.error("데이터베이스 로드 실패")
        st.error(f"❌ 데이터베이스 로드 중 오류가 발생했습니다: {e}")
        raise
