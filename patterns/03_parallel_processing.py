import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from utils import ParallelCodebase

load_dotenv()


class CodeAnalysisState(TypedDict):
    input: str
    code: str
    security_analysis: str
    performance_analysis: str
    style_analysis: str
    documented_code: str
    final_report: str


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Write ONLY Python code - no bash commands, no installation instructions, just the Python implementation."),
    ("human", "{input}")
])

security_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security Expert. Analyse code for security vulnerabilities, input validation, and potential attack vectors."),
    ("human", "Analyse this code for security issues:\n{code}")
])

performance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Performance Expert. Analyse code for efficiency, algorithmic complexity, and optimisation opportunities."),
    ("human", "Analyse this code for performance issues:\n{code}")
])

style_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Style Expert. Analyse code for PEP 8 compliance, naming conventions, and code organisation."),
    ("human", "Analyse this code for style and readability issues:\n{code}")
])

docstring_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Documentation Expert. Generate docstrings for every Python function."),
    ("human", "Generate docstrings:\n{code}")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Technical Lead. Synthesise analysis reports into actionable recommendations with priorities."),
    ("human",
     "Security Analysis:\n{security}\n\nPerformance Analysis:\n{performance}\n\nStyle Analysis:\n{style}\n\nProvide prioritised recommendations:")
])


def coder_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    return {"code": response.content}


def security_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(security_prompt.format_messages(code=state["code"]))
    return {"security_analysis": response.content}


def performance_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(
        performance_prompt.format_messages(code=state["code"]))
    return {"performance_analysis": response.content}


def style_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(style_prompt.format_messages(code=state["code"]))
    return {"style_analysis": response.content}


def doc_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(docstring_prompt.format_messages(code=state["code"]))
    return {"documented_code": response.content}


def synthesis_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(synthesis_prompt.format_messages(
        security=state["security_analysis"],
        performance=state["performance_analysis"],
        style=state["style_analysis"]
    ))
    return {"final_report": response.content}

# Parallel 
builder_parallel = StateGraph(CodeAnalysisState)
builder_parallel.add_node("coder", coder_agent)
builder_parallel.add_node("security_agent", security_agent)
builder_parallel.add_node("performance_agent", performance_agent)
builder_parallel.add_node("style_agent", style_agent)
builder_parallel.add_node("documentation", doc_agent)
builder_parallel.add_node("synthesis", synthesis_agent)

builder_parallel.add_edge(START, "coder")
builder_parallel.add_edge("coder", "security_agent")
builder_parallel.add_edge("coder", "performance_agent")
builder_parallel.add_edge("coder", "style_agent")
builder_parallel.add_edge("coder", "documentation")
builder_parallel.add_edge("security_agent", "synthesis")
builder_parallel.add_edge("performance_agent", "synthesis")
builder_parallel.add_edge("style_agent", "synthesis")
builder_parallel.add_edge("documentation", "synthesis")
builder_parallel.add_edge("synthesis", END)

workflow_parallel = builder_parallel.compile()

# Sequential 
builder_sequential = StateGraph(CodeAnalysisState)
builder_sequential.add_node("coder", coder_agent)
builder_sequential.add_node("security_agent", security_agent)
builder_sequential.add_node("performance_agent", performance_agent)
builder_sequential.add_node("style_agent", style_agent)
builder_sequential.add_node("documentation", doc_agent)
builder_sequential.add_node("synthesis", synthesis_agent)

builder_sequential.add_edge(START, "coder")
builder_sequential.add_edge("coder", "security_agent")
builder_sequential.add_edge("security_agent", "performance_agent")
builder_sequential.add_edge("performance_agent", "style_agent")
builder_sequential.add_edge("style_agent", "documentation")
builder_sequential.add_edge("documentation", "synthesis")
builder_sequential.add_edge("synthesis", END)

workflow_sequential = builder_sequential.compile()

if __name__ == "__main__":
    task = "Write a web API endpoint that processes user uploads and stores them in a database"

    print("Running sequential processing...")
    start = time.time()
    result = workflow_sequential.invoke({"input": task})
    end = time.time()
    print(f"Sequential processing took {end - start:.2f} seconds.")
    
    print("Running parallel processing...")
    start = time.time()
    result = workflow_parallel.invoke({"input": task})
    end = time.time()
    print(f"Parallel processing took {end - start:.2f} seconds.")

    codebase = ParallelCodebase("03_parallel_processing", task)
    codebase.generate(result)

    print("=== WORKFLOW COMPLETED ===")
