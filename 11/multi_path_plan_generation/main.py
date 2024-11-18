import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from passive_goal_creator.main import Goal, PassiveGoalCreator
from prompt_optimizer.main import OptimizedGoal, PromptOptimizer
from pydantic import BaseModel, Field
from response_optimizer.main import ResponseOptimizer


class TaskOption(BaseModel):
    description: str = Field(default="", description="작업 옵션 설명")


class Task(BaseModel):
    task_name: str = Field(..., description="작업 이름")
    options: list[TaskOption] = Field(
        default_factory=list,
        min_items=2,
        max_items=3,
        description="2~3개의 작업 옵션",
    )


class DecomposedTasks(BaseModel):
    values: list[Task] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="3~5개로 분해된 작업",
    )


class MultiPathPlanGenerationState(BaseModel):
    query: str = Field(..., description="사용자가 입력한 쿼리")
    optimized_goal: str = Field(default="", description="최적화된 목표")
    optimized_response: str = Field(default="", description="최적화된 응답")
    tasks: DecomposedTasks = Field(
        default_factory=DecomposedTasks,
        description="여러 옵션이 있는 작업 목록",
    )
    current_task_index: int = Field(default=0, description="현재 작업 인덱스")
    chosen_options: Annotated[list[int], operator.add] = Field(
        default_factory=list, description="각 작업에서 선택된 옵션 인덱스"
    )
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="수행된 작업의 결과"
    )
    final_output: str = Field(default="", description="최종 출력")


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "작업: 주어진 목표를 3~5개의 높은 수준의 작업으로 세분화하고, 각 작업마다 2~3개의 구체적인 옵션을 제공해야 한다.\n"
            "요구사항:\n"
            "1. 다음의 행동만으로 목표를 달성할 것. 지정된 행동 외에는 절대로 다른 행동을 하지 않는다.\n"
            "   - 인터넷을 이용해 목표를 달성하기 위한 조사를 한다.\n"
            "2. 각 상위 작업은 구체적이고 상세하게 기술되어야 하며, 단독으로 실행 및 검증이 가능한 정보를 포함해야 한다. 추상적인 표현을 포함하지 않아야 한다.\n"
            "3. 각 항목 수준 작업에 대해 2~3개의 서로 다른 접근 방식 또는 옵션을 제공해야 한다.\n"
            "4. 작업은 실행 가능한 순서대로 나열해야 한다.\n"
            "5. 작업은 한국어로 출력해야 합니다.\n\n"
            "REMEMBER: 실행할 수 없는 작업과 선택지를 절대 만들지 마십시오.\n\n"
            "목표: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})


class OptionPresenter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.configurable_fields(
            max_tokens=ConfigurableField(id="max_tokens")
        )

    def run(self, task: Task) -> int:
        task_name = task.task_name
        options = task.options

        print(f"\nタスク: {task_name}")
        for i, option in enumerate(options):
            print(f"{i + 1}. {option.description}")

        choice_prompt = ChatPromptTemplate.from_template(
            "과제: 주어진 과제와 옵션에 따라 가장 적합한 옵션을 선택해 주세요. 반드시 숫자로만 답변해 주세요.\n\n"
            "참고로, 당신은 다음과 같은 행동만 할 수 있습니다.\n"
            "- 인터넷을 이용해 목표를 달성하기 위한 조사를 한다.\n\n"
            "작업: {task_name}\n"
            "옵션:\n{options_text}\n"
            "선택 (1-{num_options}): "
        )

        options_text = "\n".join(
            f"{i+1}. {option.description}" for i, option in enumerate(options)
        )
        chain = (
            choice_prompt
            | self.llm.with_config(configurable=dict(max_tokens=1))
            | StrOutputParser()
        )
        choice_str = chain.invoke(
            {
                "task_name": task_name,
                "options_text": options_text,
                "num_options": len(options),
            }
        )
        print(f"==> 에이전트 선택: {choice_str}\n")

        return int(choice_str.strip()) - 1


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: Task, chosen_option: TaskOption) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        f"다음 작업을 수행하고 자세한 답변을 제공하십시오.:\n\n"
                        f"작업: {task.task_name}\n"
                        f"선택된 접근 방식: {chosen_option.description}\n\n"
                        f"요구사항:\n"
                        f"1. 필요에 따라 제공된 도구를 사용해야 합니다.\n"
                        f"2. 실행에 있어 철저하고 포괄적이어야 한다.\n"
                        f"3. 가능한 한 구체적인 사실과 데이터를 제공해야 한다.\n"
                        f"4. 발견한 내용을 명확하게 정리하는 것.\n",
                    )
                ]
            }
        )
        return result["messages"][-1].content


class ResultAggregator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(
        self,
        query: str,
        response_definition: str,
        tasks: list[Task],
        chosen_options: list[int],
        results: list[str],
    ) -> str:
        prompt = ChatPromptTemplate.from_template(
            "주어진 목표:\n{query}\n\n"
            "조사 결과:\n{task_results}\n\n"
            "주어진 목표에 대해 조사 결과를 사용하여 다음 지침에 따라 응답을 생성하십시오.\n"
            "{response_definition}"
        )
        task_results = self._format_task_results(tasks, chosen_options, results)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "task_results": task_results,
                "response_definition": response_definition,
            }
        )

    @staticmethod
    def _format_task_results(
        tasks: list[Task], chosen_options: list[int], results: list[str]
    ) -> str:
        task_results = ""
        for i, (task, chosen_option, result) in enumerate(
            zip(tasks, chosen_options, results)
        ):
            task_name = task.task_name
            chosen_option_desc = task.options[chosen_option].description
            task_results += f"작업 {i+1}: {task_name}\n"
            task_results += f"선택된 접근 방식: {chosen_option_desc}\n"
            task_results += f"결과: {result}\n\n"
        return task_results


class MultiPathPlanGeneration:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)
        self.response_optimizer = ResponseOptimizer(llm=self.llm)
        self.query_decomposer = QueryDecomposer(llm=self.llm)
        self.option_presenter = OptionPresenter(llm=self.llm)
        self.task_executor = TaskExecutor(llm=self.llm)
        self.result_aggregator = ResultAggregator(llm=self.llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(MultiPathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("present_options", self._present_options)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "present_options")
        graph.add_edge("present_options", "execute_task")
        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks.values),
            {True: "present_options", False: "aggregate_results"},
        )
        graph.add_edge("aggregate_results", END)

        return graph.compile()

    def _goal_setting(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        # 프롬프트 최적화
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        # 응답 최적화
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        return {
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        tasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": tasks}

    def _present_options(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = self.option_presenter.run(task=current_task)
        return {"chosen_options": [chosen_option]}

    def _execute_task(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = current_task.options[state.chosen_options[-1]]
        result = self.task_executor.run(
            task=current_task,
            chosen_option=chosen_option,
        )
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            tasks=state.tasks.values,
            chosen_options=state.chosen_options,
            results=state.results,
        )
        return {"final_output": final_output}

    def run(self, query: str) -> str:
        initial_state = MultiPathPlanGenerationState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state.get("final_output", "최종 답변 생성에 실패했습니다.")


def main():
    import argparse

    from settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="MultiPathPlanGeneration을 사용하여 작업을 수행합니다."
    )
    parser.add_argument("--task", type=str, required=True, help="수행할 작업")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    agent = MultiPathPlanGeneration(llm=llm)
    result = agent.run(query=args.task)
    print("\n=== 最終出力 ===")
    print(result)


if __name__ == "__main__":
    main()