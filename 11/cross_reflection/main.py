from common.reflection_manager import ReflectionManager, TaskReflector
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from self_reflection.main import ReflectiveAgent


def main():
    import argparse

    from settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="ReflectiveAgent를 사용하여 작업을 수행합니다（Cross-reflection）"
    )
    parser.add_argument("--task", type=str, required=True, help="수행할 작업")
    args = parser.parse_args()

    # OpenAI의 LLM 초기화
    openai_llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )

    # Anthropic의 LLM 초기화
    anthropic_llm = ChatAnthropic(
        model=settings.anthropic_smart_model, temperature=settings.temperature
    )

    # ReflectionManager 초기화
    reflection_manager = ReflectionManager(file_path="tmp/cross_reflection_db.json")

    # Anthropic의 LLM을 사용하는 TaskReflector를 초기화합니다.
    anthropic_task_reflector = TaskReflector(
        llm=anthropic_llm, reflection_manager=reflection_manager
    )

    # ReflectiveAgent 초기화
    agent = ReflectiveAgent(
        llm=openai_llm,
        reflection_manager=reflection_manager,
        task_reflector=anthropic_task_reflector,
    )

    # 작업 수행 및 결과 얻기
    result = agent.run(args.task)

    # 결과 출력
    print(result)


if __name__ == "__main__":
    main()
