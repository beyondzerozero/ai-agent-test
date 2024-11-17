import operator
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# .env파일에서 환경변수 불어오기
load_dotenv()


# 페리소나를 나타내는 데이터 모델
class Persona(BaseModel):
    name: str = Field(..., description="페르소나명")
    background: str = Field(..., description="페르소나가 가진 배경")


# 페리소나의 목록을 나타내는 데이터 모델
class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="페르소나 목록")


# 인터뷰 내용을 나타내는 데이터 모델
class Interview(BaseModel):
    persona: Persona = Field(..., description="인터뷰 대상 페르소나")
    question: str = Field(..., description="인터뷰 질문")
    answer: str = Field(..., description="인터뷰 답변")


# 인터뷰 결과 목록을 나타내는 데이터 모델
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="인터뷰 결과 목록"
    )


# 평가결과를 나타내는 데이터 모델
class EvaluationResult(BaseModel):
    reason: str = Field(..., description="판단 이유")
    is_sufficient: bool = Field(..., description="정보가 충분한지 여부")


# 요구사항 정의 생성형 AI 에이전트 상태
class InterviewState(BaseModel):
    user_request: str = Field(..., description="사용자 요청")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="생성된 페르소나 목록"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="실시된 인터뷰 목록"
    )
    requirements_doc: str = Field(default="", description="생성된 요구사항 정의")
    iteration: int = Field(default=0, description="페르소나 생성 및 인터뷰 반복 횟수")
    is_information_sufficient: bool = Field(
        default=False, description="정보가 충분한지 여부"
    )


# 페르소나를 생성하는 클래스
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 사용자 인터뷰를 위한 다양한 페르소나를 만드는 전문가입니다.",
                ),
                (
                    "human",
                    f"아래 사용자 요청에 대한 인터뷰를 위해, {self.k}사람의 다양한 페르소나를 생성해 주세요.\n\n"
                    "사용자요청: {user_request}\n\n"
                    "각 페르소나는 이름과 간단한 배경을 포함해야 합니다. 연령, 성별, 직업, 기술 전문성에서 다양성을 확보해야 합니다.",
                ),
            ]
        )
        # 페르소나 생성을 위한 체인 생성
        chain = prompt | self.llm
        # 페르소나 생성
        return chain.invoke({"user_request": user_request})


# 면접을 진행하는 클래스
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 질문 생성
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        # 답변 생성
        answers = self._generate_answers(personas=personas, questions=questions)
        # 질문과 답변의 조합으로 인터뷰 목록 작성
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )
        # 인터뷰 결과 반환
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: list[Persona]
    ) -> list[str]:
        # 질문 생성을 위한 프롬프트 정의
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 사용자 요구 사항에 따라 적절한 질문을 생성하는 전문가입니다.",
                ),
                (
                    "human",
                    "아래 페르소나와 관련된 사용자 요청에 대해 하나의 질문을 생성해 주세요.\n\n"
                    "사용자요청: {user_request}\n"
                    "페르소나: {persona_name} - {persona_background}\n\n"
                    "질문은 구체적이고, 이 페르소나의 관점에서 중요한 정보를 끌어낼 수 있도록 설계해야 합니다.",
                ),
            ]
        )
        # 질문 생성을 위한 체인 생성
        question_chain = question_prompt | self.llm | StrOutputParser()

        # 각 페르소나에 대한 질문 쿼리 생성
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        # 일괄처리로 질문 생성
        return question_chain.batch(question_queries)

    def _generate_answers(
        self, personas: list[Persona], questions: list[str]
    ) -> list[str]:
        # 답변 생성을 위한 프롬프트 정의
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 다음과 같은 페르소나로 응답하고 있습니다: {persona_name} - {persona_background}",
                ),
                ("human", "질문: {question}"),
            ]
        )
        # 답변 생성을 위한 체인 생성
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # 각 페르소나에 대한 답변 쿼리 생성
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        # 일괄처리로 답변 생성
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, personas: list[Persona], questions: list[str], answers: list[str]
    ) -> list[Interview]:
        # 각 페르소나별 질문과 답변의 조합으로 인터뷰 객체 생성
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]


# 정보 충분성을 평가하는 클래스
class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    # 사용자요청 및 인터뷰결과를 바탕으로 정보의 충분성 평가
    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        # 프롬프트 정의
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 수집된 정보를 기반으로 포괄적인 요구사항 문서를 작성하는 전문가입니다.",
                ),
                (
                    "human",
                    "아래 사용자요청과 인터뷰 결과를 바탕으로 종합적인 요구사항 문서를 작성하기에 충분한 정보가 수집되었는지 여부를 판단하십시오.\n\n"
                    "사용자요청: {user_request}\n\n"
                    "인터뷰결과:\n{interview_results}",
                ),
            ]
        )
        # 정보 충분성을 평가하는 체인 생성
        chain = prompt | self.llm
        # 평가 결과 반환
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"페르소나: {i.persona.name} - {i.persona.background}\n"
                    f"질문: {i.question}\n답변: {i.answer}\n"
                    for i in interviews
                ),
            }
        )


# 요구사항 문서 생성기 클래스
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        # 프롬프트 정의
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "수집한 정보를 바탕으로 요구사항 문서를 작성하는 전문가입니다.",
                ),
                (
                    "human",
                    "아래 사용자 요청과 여러 페르소나의 인터뷰 결과를 바탕으로 요구사항 문서를 작성해 주세요.\n\n"
                    "사용자요청: {user_request}\n\n"
                    "인터뷰결과:\n{interview_results}\n"
                    "요구사항 문서에 다음 섹션을 포함해야 합니다:\n"
                    "1. 프로젝트 개요\n"
                    "2. 주요 기능\n"
                    "3. 비기능적 요구사항\n"
                    "4. 제약조건\n"
                    "5. 타겟 사용자\n"
                    "6. 우선순위\n"
                    "7. 위험과 완화방안\n\n"
                    "출력은 반드시 한국어로 해주세요.\n\n요구사항문서:",
                ),
            ]
        )
        # 요구사항 정의서를 생성하는 체인생성
        chain = prompt | self.llm | StrOutputParser()
        # 요구사항 정의서 생성
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"페르소나: {i.persona.name} - {i.persona.background}\n"
                    f"질문: {i.question}\n답변: {i.answer}\n"
                    for i in interviews
                ),
            }
        )


# 요구사항정의서 생성형AI 에이전트 클래스
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        # 각 클래스 초기화
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)

        # 그래프 작성
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # 그래프 초기화
        workflow = StateGraph(InterviewState)

        # 각 노드 추가
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # 진입점 설정
        workflow.set_entry_point("generate_personas")

        # 노드간 엣지 추가
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)

        # 그래프 컴파일
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        # 페르소나 생성
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # 인터뷰 실시
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        # 정보 평가
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        # 요구사항정의서 생성
        requirements_doc: str = self.requirements_generator.run(
            state.user_request, state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        # 초기 상태 설정
        initial_state = InterviewState(user_request=user_request)
        # 그래프 실행
        final_state = self.graph.invoke(initial_state)
        # 최종 요구사항 정의서 획득
        return final_state["requirements_doc"]


# 실행방법:
# poetry run python -m documentation_agent.main --task "여기에 사용자요청을 입력함"
# 예제）
# poetry run python -m documentation_agent.main --task "스마트폰용 헬스케어앱을 개발하고 싶다"
def main():
    import argparse

    # 명령줄 인수 파서 생성
    parser = argparse.ArgumentParser(
        description="사용자 요청에 따라 요구사항 정의 생성"
    )
    # "task"인수추가
    parser.add_argument(
        "--task",
        type=str,
        help="만들고자 하는 애플리케이션에 대해 설명해 주세요.",
    )
    # "k"인수추가
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="생성할 페르소나 수를 설정하세요(기본값:5）",
    )
    # 명령줄 인수 파싱
    args = parser.parse_args()

    # ChatOpenAI모델 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    # 요구사항 정의서 생성 AI 에이전트 초기화
    agent = DocumentationAgent(llm=llm, k=args.k)
    # 에이전트를 실행하여 최종 결과물을 얻음
    final_output = agent.run(user_request=args.task)

    # 최종 출력 표시
    print(final_output)


if __name__ == "__main__":
    main()
