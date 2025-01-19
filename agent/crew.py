import os
import logging
import django

# Django 설정 초기화
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CANCERCAREAGENT.settings')
#django.setup()

from dotenv import load_dotenv
from crewai import Crew, Agent, Task
from crewai.process import Process
from langchain_openai import ChatOpenAI

# 상대 경로 임포트로 변경
from agent.agents import Agents
from agent.tasks import Tasks

load_dotenv()
logger = logging.getLogger('agent.crew')

def create_crew():
    logger.info("Crew 생성 시작")
    try:
        agents = Agents()
        tasks = Tasks()

        logger.info("Agent 생성 중...")

        rag_agent = agents.rag_agent()
        researcher = agents.researcher()
        health_expert = agents.health_expert()
        
        logger.info("Agent 생성 완료")  

        logger.info("Task 생성 중...")
        rag_search_task = tasks.rag_search(rag_agent)
        research_task = tasks.research(researcher)
        health_task = tasks.health_recommendation(
            health_expert,
            [rag_search_task, research_task],
        )
        logger.info("Task 생성 완료")

        crew_instance = Crew(
            agents=[rag_agent, researcher, health_expert],
            tasks=[rag_search_task, research_task, health_task],
            verbose=True,
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(model="gpt-4o",
                                   temperature=0.7,
                                   max_tokens=1500,  # 토큰 수 제한
                                   request_timeout=60),
            memory=True,

        )
        logger.info("Crew 생성 완료")
        return crew_instance

    except Exception as e:
        logger.error(f"Crew 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise

# 싱글톤 패턴으로 변경
_crew_instance = None

def get_crew():
    """
    Crew 인스턴스를 가져오거나 생성하는 싱글톤 getter
    """
    global _crew_instance
    if _crew_instance is None:
        logger.info("Crew 싱글톤 인스턴스 생성 시작")
        try:
            _crew_instance = create_crew()
            logger.info("Crew 싱글톤 인스턴스 생성 완료")
        except Exception as e:
            logger.error(f"Crew 싱글톤 인스턴스 생성 실패: {str(e)}", exc_info=True)
            raise
    return _crew_instance
