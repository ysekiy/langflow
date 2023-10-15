from langflow import CustomComponent
from langchain.tools import Tool
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.agents.agent import AgentExecutor
from langchain.agents import initialize_agent

class PlagableReactAgentComponent(CustomComponent):
    display_name: str = "Plagable React Agent"
    description: str = "Function callable Agent. You can change prompt"

    def build_config(self):
        agent = [
            "zero-shot-react-description",
            "react-docstore",
            "self-ask-with-search",
            "conversational-react-description",
            "chat-zero-shot-react-description",
            "chat-conversational-react-description",
            "structured-chat-zero-shot-react-description",
            "openai-functions",
            "openai-multi-functions"
        ]
        return {
            "llm": {"display_name": "LLM"},
            "tools": {"is_list": True, "display_name": "Tools"},
            "prefix": {"display_name": "prefix"},
            "suffix": {"display_name": "suffix"},
            "format_instructions": {"display_name": "format_instructions"},
            "agent": {
                "display_name": "Agent Type",
                "options": agent,
                "value": agent[0],
            },
            "code": {"show": False},
        }

    def build(
        self,
        llm: BaseLLM,
        tools: Tool,
        prefix: PromptTemplate,
        suffix: PromptTemplate,
        format_instructions: PromptTemplate,
        agent: str
    ) -> AgentExecutor:
        
        return initialize_agent(
            llm=llm,
            tools=tools,
            agent=agent,
            verbose=True,
            return_intermediate=True,
            agent_kwargs={
                'prefix':prefix.template,
                'format_instructions':format_instructions.template,
                'suffix':suffix.template
            }
        )


