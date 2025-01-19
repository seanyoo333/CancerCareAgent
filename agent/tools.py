import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from crewai.tools import tool
from scholarly import scholarly
import requests
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
from datetime import datetime
from database.vector_store_manager import VectorStoreManager


# 모듈 레벨에서 전역 변수 선언
current_year = datetime.now().year
class Tools:
    def __init__(self):
        """Tools 클래스 초기화"""
        pass

    @tool("RAG_refering Agent")
    def rag_search(self, query: str) -> dict:
        """
        refer to the saved retrival files for information about {query} from the user.

        Args:
            query (str): 검색할 질문 또는 키워드
            
        Returns:
            dict: 검색 결과를 포함하는 딕셔너리
        """
        
        try:
            # RAG 검색을 위한 retriever 초기화
            vector_store_manager = VectorStoreManager(
                pdf_dir="./media/pdfs",
                vector_store_dir="./media/vectors"
            )
            retriever = vector_store_manager.load_retriever()

            if not retriever:
                return {
                    "status": "error",
                    "message": "Retriever가 초기화되지 않았습니다",
                    "results": []
                }

            # retriever를 사용하여 관련 문서 검색
            documents = retriever.invoke(query)
            
            # 검색 결과 가공
            results = []
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": {
                        "source": doc.metadata.get('source', 'unknown'),
                        "page": doc.metadata.get('page', 'unknown'),
                        "score": doc.metadata.get('score', 0.0)  # 유사도 점수
                    }
                })

            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_matches": len(results),
                "summary": {
                    "sources": len(set(r['metadata']['source'] for r in results)),
                    "avg_score": sum(r['metadata']['score'] for r in results) / len(results) if results else 0
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"검색 중 오류 발생: {str(e)}",
                "query": query,
                "results": []
            }


    @tool("Web search")
    def web_search(query: str):
        """
        search the google Scholar to enchance, fact-check the answer from the RAG_refering Agent.
        """
        try:
            # 구글 스콜라 검색 수행
            search_query = scholarly.search_pubs(query)
            results = []

            # 상위 3개 결과만 가져오기
            for i in range(3):
                try:
                    publication = next(search_query)
                    results.append({
                        'title': publication.bib.get('title', ''),
                        'author': publication.bib.get('author', ''),
                        'year': publication.bib.get('year', ''),
                        'abstract': publication.bib.get('abstract', ''),
                        'url': publication.bib.get('url', ''),
                        'num_citations': publication.citedby or 0,
                        'study_type': 'unknown',  # Would need additional processing to determine
                        'journal': publication.bib.get('journal', '')
                    })
                except StopIteration:
                    break
            
            return results

        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {str(e)}"

    @tool("Web scrape")
    def web_scrape(search_results: list):
        """
        scrape the most trustworthy academic thesis or paper from the web page to get the evidence of the answer from the query, rag_search, web_search.
             
        Args: search_results: paper list from web_search
        """
        def fetch_paper_content(url):
            """Helper function to fetch and parse paper content"""
            try:
                response = requests.get(url, stream=True)  # Use streaming for large files
                if response.status_code == 200:
                    # PDF 파일인 경우
                    if url.lower().endswith('.pdf'):
                        pdf_file = BytesIO(response.content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        content = ' '.join(page.extract_text() for page in pdf_reader.pages)
                    # HTML 페이지인 경우
                    else:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # 일반적인 논문 본문 위치의 클래스나 ID를 찾아 파싱
                        content = ' '.join([p.text for p in soup.find_all(['p', 'article'])])
                    return content[:50000]  # 콘텐츠 길이 제한
                return None
            except Exception as e:
                print(f"Content fetching error: {str(e)}")
                return None
       

        # 증거 수준 평가 기준 (의학적 증거 기반)
        evidence_hierarchy = {
            'systematic review': 10,      # 체계적 문헌고찰
            'meta-analysis': 9,          # 메타분석
            'randomized controlled trial': 8,  # 무작위 대조군 연구
            'clinical guideline': 8,      # 임상 진료지침
            'cohort study': 7,           # 코호트 연구
            'clinical trial': 7,         # 임상시험
            'case-control study': 6,     # 환자-대조군 연구
            'case series': 5,            # 증례군 연구
            'case report': 4,            # 증례보고
            'expert opinion': 3,         # 전문가 의견
            'in vivo': 5,               # 생체 내 실험
            'in vitro': 4               # 시험관 내 실험
        }

        def evaluate_evidence(paper):
            """
            calculate the evidence level of the paper.
            """
            score = 0
            global current_year

            # 최신성 반영 (최근 5년 이내 논문 우대)
            year = int(paper.get('year', 0))
            if current_year - year <= 5:
                score += 5

            # 인용 수 반영
            citations = paper.get('num_citations', 0)
            score += min(citations / 50, 10)  # 최대 10점

            # peer Review 상태 반영
            if paper.get('peer_reviewed', False):
                score += 5

            # 연구 방법론 점수 반영
            study_type = paper.get('study_type', '').lower()
            for key, value in evidence_hierarchy.items():
                if key in study_type:
                    score += value
                    break
            
            return score
        
        
        try:
            # 논문들의 점수 계산 및 최고 점수 논문 선택
            scored_papers = [(paper, evaluate_evidence(paper)) for paper in search_results]
            best_paper = max(scored_papers, key=lambda x: x[1])[0]

            # 논문 전문 가져오기
            paper_content = None
            if best_paper.get('url'):
                paper_content = fetch_paper_content(best_paper['url'])

            # 결과 반환 시 paper_content 추가
            years_old = current_year - int(best_paper.get('year', current_year))
            citations = best_paper.get('num_citations', 0)
            citations_per_year = citations / max(1, years_old)

            # 연구의 신뢰도 수준 평가
            evidence_level = (
                "매우 높음" if best_paper.get('study_type') in ['systematic review', 'meta-analysis']
                else "높음" if best_paper.get('study_type') in ['randomized controlled trial', 'clinical guideline']
                else "중간" if best_paper.get('study_type') in ['cohort study', 'clinical trial', 'case-control study']
                else "제한적"
            )

            return {
                'status': 'success',
                'evidence': {
                    'citation': {
                        'title': best_paper['title'],
                        'authors': best_paper.get('authors', ''),
                        'year': best_paper.get('year', ''),
                        'journal': best_paper.get('journal', ''),
                        'url': best_paper.get('url', '')
                    },
                    'reliability': {
                        'study_type': best_paper.get('study_type', 'unknown'),
                        'evidence_level': evidence_level,
                        'citations_total': citations,
                        'citations_per_year': round(citations_per_year, 2)
                    },
                    'summary': {
                        'abstract': best_paper.get('abstract', '요약을 찾을 수 없습니다.'),
                        'key_findings': f"이 {best_paper.get('study_type', '연구')}는 {best_paper.get('year', '')}년에 발표되었으며, "
                                      f"연간 {round(citations_per_year, 1)}회 인용되는 {evidence_level} 수준의 증거를 제공합니다."
                    }
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"논문 평가 중 오류가 발생했습니다: {str(e)}",
                'fallback': "다른 신뢰할 수 있는 출처를 참고하시기 바랍니다."
            }


 
    @tool("Prescription")
    def prescription(condition: str, patient_info: dict):
        """
        Generate useful pre-formeted and well-balanced prescription for the patient through the few-shot examples.

        """
        few_shot_examples = [
            {
                "condition": "고혈압",
                "patient": {"age": 45, "gender": "M", "weight": 75},
                "prescription": """
                1. 약물 처방:
                   - Amlodipine 5mg, 1일 1회
                   - Losartan 50mg, 1일 1회
                2. 생활수칙:
                   - 저염식 식단 유지
                   - 규칙적인 운동 (주 3회 이상)
                   - 금연 및 절주
                3. 추적관찰:
                   - 2주 후 재진료
                   - 혈압 자가측정 기록
                """
            }
        ]

        template = """
        처방전
        
        진단명: {condition}
        환자정보: 나이 {age}세, 성별 {gender}
        
        1. 약물 처방:
           {medications}
        
        2. 생활수칙:
           {lifestyle}
        
        3. 추적관찰:
           {followup}
        """    
        prescription = template.format(
            condition=condition,
            age=patient_info.get('age', 'N/A'),
            gender=patient_info.get('gender', 'N/A'),
            medications="처방약 내용...",
            lifestyle="생활수칙 내용...",
            followup="추적관찰 계획..."
        )
        
        return prescription

