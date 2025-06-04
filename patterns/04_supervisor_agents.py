from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from utils import SupervisorCodebase

load_dotenv()


class SupervisorState(TypedDict):
    input: str
    code: str
    security_report: str
    quality_report: str
    database_report: str
    completed_agents: list
    final_analysis: str
    next_agent: str


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Write ONLY production-ready Python code with proper error handling - no bash commands, no installation instructions, just the Python implementation."),
    ("human", "{input}")
])

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Choose next expert: 'security_expert', 'quality_expert', or 'complete'. Completed: {completed_agents}"),
    ("human", "Code needs analysis:\n{code}")
])

security_expert_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security Expert. Focus on vulnerabilities and security best practices."),
    ("human", "Security analysis for:\n{code}")
])

quality_expert_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Quality Expert. Review code structure and maintainability."),
    ("human", "Quality analysis for:\n{code}")
])

database_expert_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Database Expert. Review database operations and maintainability."),
    ("human", "Database analysis for:\n{code}")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create final analysis summary with key recommendations."),
    ("human", "Security: {security_report}\n\nQuality: {quality_report}")
])


def coder_agent(state: SupervisorState) -> SupervisorState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    return {
        "code": response.content,
        "completed_agents": [],
        "security_report": "",
        "quality_report": "",
        "database_report": ""
    }


def supervisor_agent(state: SupervisorState) -> SupervisorState:
    completed = state.get("completed_agents", [])

    if "security_expert" not in completed:
        return {"next_agent": "security_expert"}
    elif "quality_expert" not in completed:
        return {"next_agent": "quality_expert"}
    elif "database_expert" not in completed:
        return {"next_agent": "database_expert"}
    else:
        return {"next_agent": "complete"}


def security_expert_agent(state: SupervisorState) -> SupervisorState:
    response = llm.invoke(
        security_expert_prompt.format_messages(code=state["code"]))
    completed = state.get("completed_agents", []).copy()
    completed.append("security_expert")
    return {"security_report": response.content, "completed_agents": completed}


def quality_expert_agent(state: SupervisorState) -> SupervisorState:
    response = llm.invoke(
        quality_expert_prompt.format_messages(code=state["code"]))
    completed = state.get("completed_agents", []).copy()
    completed.append("quality_expert")
    return {"quality_report": response.content, "completed_agents": completed}


def database_expert_agent(state: SupervisorState) -> SupervisorState:
    response = llm.invoke(
        database_expert_prompt.format_messages(code=state["code"]))
    completed = state.get("completed_agents", []).copy()
    completed.append("database_expert")
    return {"database_report": response.content, "completed_agents": completed}


def synthesis_agent(state: SupervisorState) -> SupervisorState:
    response = llm.invoke(synthesis_prompt.format_messages(
        security_report=state.get("security_report", "Not analysed"),
        quality_report=state.get("quality_report", "Not analysed"),
        database_report=state.get("database_report", "Not analysed")
    ))
    return {"final_analysis": response.content}


def route_to_expert(state: SupervisorState) -> Literal["security_expert", "quality_expert", "database_expert", "synthesis"]:
    next_agent = state.get("next_agent", "complete")
    if next_agent == "complete":
        return "synthesis"
    return next_agent


builder = StateGraph(SupervisorState)
builder.add_node("coder", coder_agent)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("security_expert", security_expert_agent)
builder.add_node("quality_expert", quality_expert_agent)
builder.add_node("database_expert", database_expert_agent)
builder.add_node("synthesis", synthesis_agent)

builder.add_edge(START, "coder")
builder.add_edge("coder", "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_expert,
    {
        "security_expert": "security_expert",
        "quality_expert": "quality_expert",
        "database_expert": "database_expert",
        "synthesis": "synthesis"
    }
)
builder.add_edge("security_expert", "supervisor")
builder.add_edge("quality_expert", "supervisor")
builder.add_edge("database_expert", "supervisor")
builder.add_edge("synthesis", END)

workflow = builder.compile()

if __name__ == "__main__":
    task = "Write a user authentication system"

    print("Starting supervised code review...")
    result = workflow.invoke({"input": task})

    codebase = SupervisorCodebase("04_supervisor_agents", task)
    codebase.generate(result)

    print("=== WORKFLOW COMPLETED ===")
