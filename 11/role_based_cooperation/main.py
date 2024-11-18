import operator
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from single_path_plan_generation.main import DecomposedTasks, QueryDecomposer


class Role(BaseModel):
    name: str = Field(..., description="역할 이름")
    description: str = Field(..., description="역할 상세 설명")
    key_skills: list[str] = Field(..., description="이 역할에 필요한 주요 기술 및 특성")


class Task(BaseModel):
    description: str = Field(..., description="작업 설명")
    role: Role = Field(default=None, description="작업에 할당된 역할")


class TasksWithRoles(BaseModel):
    tasks: list[Task] = Field(..., description="역할이 할당된 작업 목록")


class AgentState(BaseModel):
    query: str = Field(..., description="사용자가 입력한 쿼리")
    tasks: list[Task] = Field(
        default_factory=list, description="실행할 작업 목록"
    )
    current_task_index: int = Field(default=0, description="현재 실행 중인 작업 수")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="실행된 작업 결과 목록"
    )
    final_report: str = Field(default="", description="최종 출력 결과")


class Planner:
    def __init__(self, llm: ChatOpenAI):
        self.query_decomposer = QueryDecomposer(llm=llm)

    def run(self, query: str) -> list[Task]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(query=query)
        return [Task(description=task) for task in decomposed_tasks.values]


class RoleAssigner:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(TasksWithRoles)

    def run(self, tasks: list[Task]) -> list[Task]:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "당신은 창의적인 역할 설계 전문가입니다. 주어진 업무에 대해 독특하고 적절한 역할을 생성하세요."
                    ),
                ),
                (
                    "human",
                    (
                        "작업:\n{tasks}\n\n"
                        "다음 지침에 따라 이러한 작업에 대한 역할을 할당하십시오：\n"
                        "1. 각 업무에 대해 자신만의 창의적인 역할을 만들어 보세요. 기존의 직업명이나 일반적인 역할 이름에 얽매일 필요는 없습니다.\n"
                        "2. 역할명은 해당 업무의 본질을 반영하는 매력적이고 기억에 남는 이름이어야 합니다.\n"
                        "3. 각 역할에 대해 해당 역할이 왜 그 업무에 가장 적합한지를 설명할 수 있는 자세한 설명을 제공해야 합니다.\n"
                        "4. 해당 역할이 효과적으로 업무를 수행하기 위해 필요한 주요 기술이나 특성을 3가지만 꼽아주세요.\n\n"
                        "창의력을 발휘하여 업무의 본질을 파악한 혁신적인 역할을 창출하세요."
                    ),
                ),
            ],
        )
        chain = prompt | self.llm
        tasks_with_roles = chain.invoke(
            {"tasks": "\n".join([task.description for task in tasks])}
        )
        return tasks_with_roles.tasks


class Executor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]
        self.base_agent = create_react_agent(self.llm, self.tools)

    def run(self, task: Task) -> str:
        result = self.base_agent.invoke(
            {
                "messages": [
                    (
                        "system",
                        (
                            f"당신은 {task.role.name}입니다.\n"
                            f"설명: {task.role.description}\n"
                            f"주요 기술: {', '.join(task.role.key_skills)}\n"
                            "자신의 역할에 따라 주어진 업무를 최선을 다해 수행하세요."
                        ),
                    ),
                    (
                        "human",
                        f"다음 작업을 수행하십시오：\n\n{task.description}",
                    ),
                ]
            }
        )
        return result["messages"][-1].content


class Reporter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, results: list[str]) -> str:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "당신은 종합적인 보고서 작성 전문가입니다. 여러 출처의 결과를 통합하여 통찰력 있고 포괄적인 보고서를 작성할 수 있는 능력이 있습니다."
                    ),
                ),
                (
                    "human",
                    (
                        "과제: 다음 정보를 바탕으로 포괄적이고 일관성 있는 답변을 작성하세요.\n"
                        "요구사항:\n"
                        "1. 제공된 모든 정보를 통합하여 잘 짜여진 답변을 작성해 주세요.\n"
                        "2. 답변은 원래의 쿼리에 직접적으로 응답하는 형태로 작성해야 합니다.\n"
                        "3. 각 정보의 중요한 포인트와 발견을 포함하세요.\n"
                        "4. 마지막으로 결론이나 요약을 제공하십시오.\n"
                        "5. 답변은 상세하면서도 간결하게 작성하고, 250~300단어 정도로 작성하는 것이 좋습니다.\n"
                        "6. 답변은 한국어로 해주세요.\n\n"
                        "사용자 요청: {query}\n\n"
                        "수집된 정보:\n{results}"
                    ),
                ),
            ],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": "\n\n".join(
                    f"Info {i+1}:\n{result}" for i, result in enumerate(results)
                ),
            }
        )


class RoleBasedCooperation:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.planner = Planner(llm=llm)
        self.role_assigner = RoleAssigner(llm=llm)
        self.executor = Executor(llm=llm)
        self.reporter = Reporter(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self._plan_tasks)
        workflow.add_node("role_assigner", self._assign_roles)
        workflow.add_node("executor", self._execute_task)
        workflow.add_node("reporter", self._generate_report)

        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "role_assigner")
        workflow.add_edge("role_assigner", "executor")
        workflow.add_conditional_edges(
            "executor",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "executor", False: "reporter"},
        )

        workflow.add_edge("reporter", END)

        return workflow.compile()

    def _plan_tasks(self, state: AgentState) -> dict[str, Any]:
        tasks = self.planner.run(query=state.query)
        return {"tasks": tasks}

    def _assign_roles(self, state: AgentState) -> dict[str, Any]:
        tasks_with_roles = self.role_assigner.run(tasks=state.tasks)
        return {"tasks": tasks_with_roles}

    def _execute_task(self, state: AgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _generate_report(self, state: AgentState) -> dict[str, Any]:
        report = self.reporter.run(query=state.query, results=state.results)
        return {"final_report": report}

    def run(self, query: str) -> str:
        initial_state = AgentState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state["final_report"]


def main():
    import argparse

    from settings import Settings

    settings = Settings()
    parser = argparse.ArgumentParser(
        description="RoleBasedCooperation를 사용하여 작업을 수행합니다"
    )
    parser.add_argument("--task", type=str, required=True, help="수행할 작업")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    agent = RoleBasedCooperation(llm=llm)
    result = agent.run(query=args.task)
    print(result)


if __name__ == "__main__":
    main()
