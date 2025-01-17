from crewai import Crew
from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI
from .agents import Agents
from .tasks import Tasks

def create_crew():
    agents = Agents()
    tasks = Tasks()

    rag_agent = agents.rag_agent()
    researcher = agents.researcher()
    health_expert = agents.health_expert()

    rag_search_task = tasks.rag_search(rag_agent)
    research_task = tasks.research(researcher)
    health_task = tasks.health_recommendation(
        health_expert,
        [rag_search_task, research_task],
    )

    return Crew(
        agents=[rag_agent, researcher, health_expert],
        tasks=[rag_search_task, research_task, health_task],
        verbose=True,
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4o"),
        memory=True,
    )


# 싱글톤으로 인스턴스를 생성해서 API 요청마다 새로운 crew를 생성하지 않도록 함 -> 메모리 사용 최적화
crew = create_crew()
