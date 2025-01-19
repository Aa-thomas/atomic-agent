import instructor
import openai
from dotenv import load_dotenv
from atomic_agents.agents.base_agent import (
    BaseAgent,
    BaseAgentConfig,
    BaseAgentInputSchema,
    BaseAgentOutputSchema,
)
from rich.console import Console
from rich.text import Text
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

# Load environment variables
load_dotenv()

# Initialize Rich Console
console = Console()

# Initialize OpenAI client
client = instructor.from_openai(openai.OpenAI())

# Initialize BaseAgent
system_prompt_generator_custom = SystemPromptGenerator(
    background=["You are a helpful assistant that can only give facts about jamaica"],
    steps=[
        "analyze the user's question to see if its about jamaica",
        "if it is, give a fact about jamaica",
        "if it is not, say you can only give facts about jamaica",
    ],
    output_instructions=[
        "Your output should always be presented as a rhyme",
    ],
)
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=system_prompt_generator_custom,
    )
)

# Add initial message to the agent memory and print
initial_message = "Hello, how are you?"
agent.memory.add_message(
    "assistant",
    content=BaseAgentOutputSchema(chat_message=initial_message),
)


# Main Loop
if __name__ == "__main__":
    while True:
        try:
            user_input = console.input(Text("You: ", style="bold blue"))
            response = agent.run(
                BaseAgentInputSchema(
                    chat_message=user_input,
                )
            )
            console.print(
                Text(
                    f"Assistant: {response.chat_message}",
                    style="bold green",
                )
            )
            if user_input.lower() in ["exit", "quit", "q"]:
                break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break
