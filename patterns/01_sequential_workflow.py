import json
import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from utils import SequentialCodebase
from datetime import datetime
from functools import wraps

LOG_FILENAME = f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

load_dotenv()


class CodeReviewState(TypedDict):
    input: str
    code: str
    review: str
    refactored_code: str
    unit_tests: str


def log_runtime(func):
    def wrapper(state: CodeReviewState) -> CodeReviewState:
        start = time.time()
        result = func(state)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds.")
        return result
    return wrapper

def log_state(func):
    @wraps(func)
    def wrapper(state: dict) -> dict:
        result = func(state)
        result["agent"] = func.__name__

        # Create a logs directory if it doesn't exist
        os.makedirs("state_logs", exist_ok=True)

        # Create filename with timestamp and agent name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"state_logs/{timestamp}_{func.__name__}.json"

        # Save result state to the file
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

        return result
    return wrapper


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Write clean, well-structured Python code based on requirements. Pay special attention to risks of security vulnerabilities."),
    ("human", "{input}")
])

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Reviewer. Provide constructive feedback focusing on readability, efficiency, and best practices. Pay special attention to risks of security vulnerabilities."),
    ("human", "Review this code:\n{code}")
])

refactorer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Refactoring Expert. Implement the suggested improvements while maintaining functionality. Pay special attention to risks of security vulnerabilities."),
    ("human", "Original code:\n{code}\n\nReview feedback:\n{review}\n\nRefactor accordingly:")
])

tester_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Testing Expert. Generate a set of unit tests including edge cases. Pay special attention to risks of security vulnerabilities."),
    ("human", "Generate unit tests for this code:\n{refactored_code}")
])

@log_runtime
@log_state
def coder_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    return {**state, "code": response.content}

@log_runtime
@log_state
def reviewer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(reviewer_prompt.format_messages(code=state["code"]))
    return {**state, "review": response.content}

@log_runtime
@log_state
def refactorer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(refactorer_prompt.format_messages(
        code=state["code"], review=state["review"]))
    return {**state, "refactored_code": response.content}

@log_runtime
@log_state
def tester_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(tester_prompt.format_messages(refactored_code=state["refactored_code"]))
    return {**state, "unit_tests": response.content}


builder = StateGraph(CodeReviewState)
builder.add_node("coder", coder_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("refactorer", refactorer_agent)
builder.add_node("tester", tester_agent)

builder.add_edge(START, "coder")
builder.add_edge("coder", "reviewer")
builder.add_edge("reviewer", "refactorer")
builder.add_edge("refactorer", "tester")
builder.add_edge("tester", END)

workflow = builder.compile()

if __name__ == "__main__":
    task = "Write a function that validates email addresses using regex"

    print("Running sequential workflow...")
    result = workflow.invoke({"input": task})

    codebase = SequentialCodebase("01_sequential_workflow", task)
    codebase.generate(result)

    print("=== WORKFLOW COMPLETED ===")
