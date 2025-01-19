from crewai import Task
from pydantic import BaseModel, Field
from typing import Optional, List


# Task 출력을 위한 Pydantic 모델 정의
class SearchResult(BaseModel):
    summary: str = Field(description="Summary of the retrieved information")
    sources: List[str] = Field(default_factory=list, description="List of sources")

class ResearchResult(BaseModel):
    summary: str = Field(description="Research findings summary")
    evidence: List[str] = Field(default_factory=list, description="Evidence links")
    citations: List[str] = Field(default_factory=list, description="Citations")

class HealthRecommendation(BaseModel):
    recommendation: str = Field(description="Health recommendation details")
    rationale: str = Field(description="Reasoning behind the recommendation")
    references: List[str] = Field(default_factory=list, description="Reference materials")


class Tasks:
    def rag_search(self, agent):
        return Task(
            name="rag_search",
            description="refer to the saved retrival files for information about {question}.",
            expected_output="Your final answer MUST be a detailed summary of the most relevant information from the retrival files that could help your patient make informed health decisions.",
            agent=agent,
            output_json=SearchResult
        )


    def research(self, agent):
        return Task(
            name="research",
            description="search the web to enchance, fact-check the answer from the RAG_refering Agent and get the related evidence link and summary of it.",
            expected_output="Your final answer MUST be a detailed summary of the most relevant information from the retrival files that could help your patient make informed health decisions.",
            agent=agent,
            output_json=ResearchResult
        )

   
    def health_recommendation(self, agent, dependent_tasks):
        return Task(
            name="health_recommendation",
            description="Analyze {question}'s health information and provide a detailed health recommendation for the patient. Provide insights on the patient's health, treatment options, and other key health metrics.",
            expected_output="Your final answer MUST be a detailed health recommendation report that includes insights on the patient's health, treatment options, and other key health metrics.",
            agent=agent,
            dependent_tasks=dependent_tasks,
            output_json=HealthRecommendation
        )
