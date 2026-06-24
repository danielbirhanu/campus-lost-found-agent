from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

LOST_FOUND_DATABASE = [
    {
        "id": 1,
        "type": "found",
        "item": "black wallet",
        "location": "library second floor",
        "date": "Monday",
        "details": "contains student ID and small cash"
    },
    {
        "id": 2,
        "type": "found",
        "item": "blue water bottle",
        "location": "computer lab",
        "date": "Tuesday",
        "details": "metal bottle with sticker"
    },
    {
        "id": 3,
        "type": "lost",
        "item": "silver laptop charger",
        "location": "cafeteria",
        "date": "Wednesday",
        "details": "HP charger"
    },
    {
        "id": 4,
        "type": "found",
        "item": "white earbuds case",
        "location": "main gate",
        "date": "Friday",
        "details": "empty wireless earbuds case"
    }
]


class AgentState(TypedDict):
    user_input: str
    intake_summary: str
    category: str
    possible_matches: List[Dict[str, Any]]
    followup_suggestion: str
    final_answer: str
    evaluation: str


def intake_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are the Intake Agent for a campus lost-and-found system.

Extract the important information from this user message:
"{state['user_input']}"

Return a short summary including:
- whether the user lost or found something
- item description
- location
- time/date if mentioned
- any identifying detail
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "intake_summary": response.content
    }


def category_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are the Category Agent.

Classify this lost/found report into ONE category only:

Categories:
- electronics
- clothing
- ID/document
- bottle/container
- bag/wallet
- keys
- other

Report:
{state['intake_summary']}

Return only the category name.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "category": response.content.strip()
    }


def matching_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are the Matching Agent.

User report:
{state['intake_summary']}

Database records:
{LOST_FOUND_DATABASE}

Find records that may match the user's report.
Consider item name, location, date, color, and details.

Return only the matching record IDs as comma-separated numbers.
If there is no match, return "none".
"""

    response = llm.invoke(prompt)
    raw_ids = response.content.strip().lower()

    matches = []

    if raw_ids != "none":
        for record in LOST_FOUND_DATABASE:
            if str(record["id"]) in raw_ids:
                matches.append(record)

    return {
        **state,
        "possible_matches": matches
    }


def match_router(state: AgentState) -> str:
    if len(state["possible_matches"]) > 0:
        return "match_found"
    return "no_match"


def followup_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are the Follow-up Agent for a campus lost-and-found system.

The system found no matching item.

User report:
{state['intake_summary']}

Suggest what extra information the user should provide to improve the search.

Examples:
- more exact location
- date/time
- brand
- color
- unique marks
- photo if available

Write 2-3 helpful questions.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "followup_suggestion": response.content
    }


def response_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are the Response Agent for a campus lost-and-found assistant.

User message:
{state['user_input']}

Intake summary:
{state['intake_summary']}

Category:
{state['category']}

Possible matches:
{state['possible_matches']}

Follow-up suggestion:
{state['followup_suggestion']}

Write a clear final response to the user.

Rules:
- If there are matches, explain them clearly.
- If there are no matches, use the follow-up suggestion.
- Be polite and helpful.
- Mention that users must verify ownership before claiming an item.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content
    }


def evaluator_agent(state: AgentState) -> AgentState:
    prompt = f"""
You are the Evaluator Agent.

Check whether this final answer is clear, useful, and safe:

Final answer:
{state['final_answer']}

Return:
APPROVED - if the answer is good.
REVISE - if the answer needs improvement.

Also give one short reason.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "evaluation": response.content
    }


graph_builder = StateGraph(AgentState)

graph_builder.add_node("intake_agent", intake_agent)
graph_builder.add_node("category_agent", category_agent)
graph_builder.add_node("matching_agent", matching_agent)
graph_builder.add_node("followup_agent", followup_agent)
graph_builder.add_node("response_agent", response_agent)
graph_builder.add_node("evaluator_agent", evaluator_agent)

graph_builder.add_edge(START, "intake_agent")
graph_builder.add_edge("intake_agent", "category_agent")
graph_builder.add_edge("category_agent", "matching_agent")

graph_builder.add_conditional_edges(
    "matching_agent",
    match_router,
    {
        "match_found": "response_agent",
        "no_match": "followup_agent"
    }
)

graph_builder.add_edge("followup_agent", "response_agent")
graph_builder.add_edge("response_agent", "evaluator_agent")
graph_builder.add_edge("evaluator_agent", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    print("Campus Lost-and-Found Multi-Agent System")
    print("---------------------------------------")

    user_input = input("Describe your lost or found item: ")

    initial_state = {
        "user_input": user_input,
        "intake_summary": "",
        "category": "",
        "possible_matches": [],
        "followup_suggestion": "",
        "final_answer": "",
        "evaluation": ""
    }

    result = graph.invoke(initial_state)

    print("\n--- Intake Summary ---")
    print(result["intake_summary"])

    print("\n--- Category ---")
    print(result["category"])

    print("\n--- Possible Matches ---")
    print(result["possible_matches"])

    print("\n--- Follow-up Suggestion ---")
    print(result["followup_suggestion"])

    print("\n--- Final Answer ---")
    print(result["final_answer"])

    print("\n--- Evaluation ---")
    print(result["evaluation"])