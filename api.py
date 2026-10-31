from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

import train


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 시작될 때 두 모델을 한 번씩 학습
    pairs, labels = train.load_pairs()
    texts, _ = train.load_data()

    app.state.context_model = train.build_model().fit(pairs, labels)
    app.state.text_model = train.build_text_only_model().fit(texts, labels)

    print(f"학습 완료: {len(labels)}개")
    yield


app = FastAPI(lifespan=lifespan)


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    situation: str = Field(default="", max_length=5000)

    @field_validator("text")
    @classmethod
    def check_text(cls, value):
        if not value.strip():
            raise ValueError("분석할 문장을 입력하세요.")
        return value.strip()


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    situation = request.situation.strip()

    # 상황 설명이 있는지에 따라 모델 선택
    if situation:
        model = app.state.context_model
    else:
        model = app.state.text_model

    judgment, reason = train.predict_case(
        model,
        situation,
        request.text
    )

    return {
        "judgment": judgment,
        "reason": reason,
        "emergency": judgment == "긴급 위협 의심"
    }