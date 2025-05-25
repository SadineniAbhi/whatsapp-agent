import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, Sequence, TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

async def main():
    model = ChatOpenAI(model='gpt-4o', api_key='')

    client = MultiServerMCPClient({
        "whatsapp": {
            "command": "/home/abhi/whatsapp-mcp/whatsapp-mcp-server/.venv/bin/python3",
            "args": ["/home/abhi/whatsapp-mcp/whatsapp-mcp-server/main.py"],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()  


    def get_query(state: AgentState) -> AgentState:
        query = input("> ")
        return {"messages": [HumanMessage(content=query)]}

    async def call_model(state: AgentState) -> AgentState:
        response = await model.bind_tools(tools).ainvoke(state["messages"])
        print(response.content)
        return {"messages": response}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "continue"
        else:
            return "exit"

    builder = StateGraph(AgentState)
    builder.add_node("get_query", get_query)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.set_entry_point("get_query")
    builder.add_edge("get_query", "agent")
    builder.add_conditional_edges("agent", should_continue, {
        "continue": "tools",
        "exit": END
    })
    builder.add_edge("tools", "agent")
    app = builder.compile()
    initial_state = {
        "messages": [
            SystemMessage(content="You are my assistant. You help me send whatsapp message before sending you always try to understand the context before sending the messages")
        ]
    }

    while True:
        result = await app.ainvoke(initial_state)
        initial_state = result  

if __name__ == "__main__":
    asyncio.run(main())
