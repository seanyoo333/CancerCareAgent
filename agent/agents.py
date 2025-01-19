from crewai import Agent
from agent.tools import Tools

class Agents:
    def rag_agent(self):
        return Agent(
            role="RAG_refering Agent",
            goal="Refer to the saved retrival files for information about health related questions from the user.",
            backstory="You're an expert in refering to the saved retrival files and finding the most related information to the questions from the user.",
            verbose=True,
            tools=[
                Tools.rag_search,
            ],
        )

    def researcher(self):
        return Agent(
            role="Researcher",
            goal="search the web to enchance, fact-check the answer from the RAG_refering Agent and get the related evidence link and summary of it.",
            backstory="You're skilled in gathering and interpreting data from various sources to give a complete and well-balanced response to the user's question. You read each data source carefuly and extract the most important information.",
            verbose=True,
            tools=[
                Tools.web_search,
                Tools.web_scrape,
            ],
        )

    def health_expert(self):
        return Agent(
            role="Health Expert",
            goal="Use the information from the RAG_refering Agent and the Researcher to provide a comprehensive and well-balanced response to the user's question. If {prescription} is Yes, you should use the prescription tool.",
            backstory="You're a very experienced health care advisor who uses a combination of nutritional and integrative medicinal methods to provide strategic health advice to your clients. Your insights are crucial for making informed health decisions for cancer patients.",
            verbose=True,
            tools=[
                Tools.prescription,
            ],
        )
