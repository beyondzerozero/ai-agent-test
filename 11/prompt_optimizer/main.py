from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from passive_goal_creator.main import Goal, PassiveGoalCreator
from pydantic import BaseModel, Field


class OptimizedGoal(BaseModel):
    description: str = Field(..., description="목표 설명")
    metrics: str = Field(..., description="목표 달성도를 측정하는 방법")

    @property
    def text(self) -> str:
        return f"{self.description}(측정 기준: {self.metrics})"


class PromptOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            "당신은 목표 설정 전문가입니다. 다음 목표를 SMART 원칙(Specific: 구체적, Measurable: 측정 가능, Achievable: 달성 가능, Relevant: 관련성 높음, Time-bound: 기한이 정해져 있음)을 기준으로 최적화해야 한다.\n\n"
            "원래목표:\n"
            "{query}\n\n"
            "지시사항:\n"
            "1. 원래의 목표를 분석하고 부족한 요소와 개선점을 파악하세요.\n"
            "2. 당신이 할 수 있는 행동은 다음과 같은 행동뿐입니다.\n"
            "   - 인터넷을 이용해 목표를 달성하기 위한 조사를 한다.\n"
            "   - 사용자를 위한 보고서를 생성합니다.\n"
            "3. SMART 원칙의 각 요소를 고려하여 목표를 구체적이고 상세하게 기술하십시오.\n"
            "   - 추상적인 표현이 포함되어서는 안 됩니다.\n"
            "   - 모든 단어가 실행 가능하고 구체적이어야 합니다.\n"
            "4. 목표 달성도를 측정하는 방법을 구체적이고 상세하게 기술하십시오.\n"
            "5. 원래 목표에 기한이 지정되지 않은 경우 기한을 고려할 필요가 없습니다.\n"
            "6. REMEMBER: 절대로 2. 이외의 행동을 해서는 안 됩니다."
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query})


def main():
    import argparse

    from settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="PromptOptimizer를 활용하여 생성된 목표 목록을 최적화합니다"
    )
    parser.add_argument("--task", type=str, required=True, help="수행할 작업")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )

    passive_goal_creator = PassiveGoalCreator(llm=llm)
    goal: Goal = passive_goal_creator.run(query=args.task)

    prompt_optimizer = PromptOptimizer(llm=llm)
    optimised_goal: OptimizedGoal = prompt_optimizer.run(query=goal.text)

    print(f"{optimised_goal.text}")


if __name__ == "__main__":
    main()
