import os
import requests
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool, RunContextWrapper
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

class Acount(BaseModel):
    name: str
    pin: int

def check_user(ctx:RunContextWrapper[Acount], agent: Agent) -> bool:
    if ctx.context.name == "Umer Ali" and ctx.context.pin == 1234:
     return True
    else:
       return False

@function_tool(is_enabled=check_user)
def check_balance(acount_number: str) -> str:
    return f"The Balance of acount is $1000000"

bank_agent = Agent(
    name="Bank Agent",
    instructions="You are a bank agent answer the quries of the custome related to bank accounts and their balance information.",
    tools=[check_balance]
)

user_Context = Acount(
    name="Umer ALi",
    pin=1234
)

result = Runner.run_sync(
    bank_agent,
    "What is the balance of my account, Acountnumber is 12400?",
    run_config=run_config,
    context=user_Context,
)

print(result.final_output)



# Financial and Bank Related Agent