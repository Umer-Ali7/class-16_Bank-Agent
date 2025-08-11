import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool, handoff, RunContextWrapper
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash",
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Input Guardrails
class TransactionRequest(BaseModel):
    account_number: str = Field(..., pattern=r"^\d{8,12}$", description="Valid bank account number")
    amount: float = Field(0, ge=0, description="Amount must be zero or positive")
    query: str = Field(..., description="User's query about account or loan")

# Output Guardrails
class BankResponse(BaseModel):
    message: str = Field(..., max_length=200, description="Short, clear bank response")


# User Context Model
class Account(BaseModel):
    name: str
    pin: int

# Guardrail: Verify user
def check_user(ctx: RunContextWrapper[Account], agent: Agent) -> bool:
    return ctx.context.name == "Umer Ali" and ctx.context.pin == 1234


# Account Agent
@function_tool(is_enabled=check_user)
def check_balance(account_number: str) -> str:
    fake_db = {"12345678": 1000000, "87654321": 5000000}
    return f"The balance for account {account_number} is ${fake_db.get(account_number, 'Account not found')}."


account_agent = Agent(
    name="Account Agent",
    instructions="You are a account services agent. Handle balance checks and account-related queries.",
    tools=[check_balance],
    output_type=BankResponse
)

# Loan Agent
@function_tool(is_enabled=check_user)
def apply_loan(account_number: str, amount: float) -> str:
    if amount > 100000:
        return "Loan request exceeds the allowed limit."
    return f"Loan request of ${amount} for account {account_number} has been submitted."

loan_agent = Agent(
    name="Loan Agent",
    instructions="You are a loan services agent. Handle loan applications and related queries.",
    tools=[apply_loan],
    output_type=BankResponse
)

# Main Bank Agent (Triage)
bank_agent = Agent(
    name="Bank Agent",
    instructions="""
    You are the main bank triage agent for GenZ Bank.
    1. Validate the input.
    2. If the query is about 'loan', handoff to the Loan Agent.
    3. If the query is about 'balance' or 'account', handoff to the Account Agent.
    4. Always keep responses short and clear.
    """,
    input_guardrails=TransactionRequest,
    output_type=BankResponse,
    handoffs=[account_agent, loan_agent]
)

# @bank_agent.tools
# def route_request(query: str) -> handoff:
#     q = query.lower()
#     if "loan" in q:
#         return handoff(to="Loan Agent")
#     elif "balance" in q or "account" in q:
#         return handoff(to="Account Agent")
#     else:
#         return BankResponse(message="Sorry, I cannot process that request.")

# Runner setup
# runner = Runner(
# #     agents=[bank_agent, account_agent, loan_agent],
# #     run_config=run_config,
# )

# Simulation
if __name__ == "__main__":
    user_context = Account(name="Umer Ali", pin=1234)

    print("🏦 Welcome to GenZ Bank Agent System!")
    while True:
        user_query = input("💬 You are query? (Type 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # For testing we provide a valid account number & amount here\
        input_data = {
            "account_number": "12345678",
            "amount": 5000,
            "query": user_query
        }

        result = Runner.run_sync(bank_agent, input=input_data, context=user_context, )
        print(f"🤖 Bank Agent: {result.final_output}")