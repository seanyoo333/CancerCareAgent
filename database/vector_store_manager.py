from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import shutil
from datetime import datetime

class VectorStoreManager:
    def __init__(self, pdf_dir: str, vector_store_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.backup_dir = self. vector_store_dir / "backups"
        self.embeddings = OpenAIEmbeddings()

    def create_or_update(self):
        """백터 저장소 생성 또는 업데이트"""

        # 백업 디렉토리 생성
        self.backup_dir.mkdir(exist_ok=True)

        # 현재 백터 자장소 백업
        if self.vector_store_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = self.backup_dir / f"vectorstore_backup_{timestamp}"
            shutil.copytree(self.vector_store_dir, backup_path, dirs_exist_ok=True)
            

        # pdf 문서 로드
        documents = []
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        # 문서분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = splitter.split_documents(documents)

        # 벡터 저장소 생성
        vectorstore = FAISS.from_documents(splits, self.embeddings)

        # 저장
        vectorstore.save_local(str(self.vector_store_dir))
        print(f"Vector store updated with {len(splits)} chunks")


