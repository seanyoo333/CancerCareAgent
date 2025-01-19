from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

class VectorStoreManager:
    def __init__(
        self,
        pdf_dir: str,
        vector_store_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 50
    ):
        """PDF 문서를 벡터화하고 검색 가능한 형태로 관리"""
        self.pdf_dir = Path("./media/pdfs")
        self.vector_store_dir = Path("./media/vectors")
        self.backup_dir = self.vector_store_dir / "backups"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()

    def create_or_update(self) -> Optional[BaseRetriever]:
        """벡터 저장소 생성 또는 업데이트"""
        try:
            # 기존 저장소 백업
            if self.vector_store_dir.exists():
                self.backup_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                backup_path = self.backup_dir / f"vectorstore_backup_{timestamp}"
                shutil.copytree(self.vector_store_dir, backup_path, dirs_exist_ok=True)

            # PDF 문서 로드 및 분할
            documents = []
            for pdf_file in self.pdf_dir.glob("*.pdf"):
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())

            if not documents:
                print("처리할 PDF 문서가 없습니다.")
                return None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            splits = splitter.split_documents(documents)

            # 벡터 저장소 생성 및 저장
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            vectorstore.save_local(str(self.vector_store_dir))
            
            # retriever 생성 및 반환
            return vectorstore.as_retriever(
                search_type="similarity",  # similarity, mmr 중 선택 가능
                search_kwargs={
                    "k": 4,  # 검색할 문서 수
                    # "score_threshold": 0.5,  # 유사도 임계값 (선택사항)
                    # "fetch_k": 20  # MMR 검색시 후보 문서 수 (선택사항)
                }
            )

        except Exception as e:
            print(f"벡터 저장소 생성/업데이트 실패: {e}")
            return None

    def load_retriever(self) -> Optional[BaseRetriever]:
        """저장된 벡터 저장소로부터 retriever 로드"""
        try:
            if not self.vector_store_dir.exists():
                print("벡터 저장소가 존재하지 않습니다.")
                return None

            vectorstore = FAISS.load_local(
                str(self.vector_store_dir), 
                self.embeddings
            )
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4,
                }
            )

        except Exception as e:
            print(f"retriever 로드 실패: {e}")
            return None


