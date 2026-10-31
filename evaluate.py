"""학습에 넣지 않은 예시 18개로 현재 낌새 판정 모델의 오류를 살펴본다.

사용법: python evaluate.py
이 예시들은 직접 작성한 작은 시험 자료다. 서비스 성능을 뜻하지 않는다.
오답을 보고 학습 데이터에 이 문장들을 옮기면 독립 시험이 아니게 된다.
"""

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from train import format_input, load_data


CASES = [
    ("고객센터 직원이라며 결제 취소를 도와주겠다고 했어. 문자로 받은 일회용 번호를 불러 달래.", "사기 의심"),
    ("면접 합격이라더니, 근로계약서를 보내기 전에 보증금을 개인 계좌로 보내래.", "사기 의심"),
    ("중고 책을 산다고 하니까 판매 앱에서는 취소하고 그 사람이 준 주소에서 카드로 결제하래.", "사기 의심"),
    ("여행 중인 애인이 갑자기 계좌가 동결됐다며 귀국 항공권 값을 보내 달라고 해.", "사기 의심"),
    ("택배 반송을 막으려면 문자에 첨부된 파일을 설치하고 휴대폰 권한을 허용하래.", "사기 의심"),
    ("부업 수익을 받으려면 먼저 내 돈으로 미션 상품을 구매해야 한다고 해.", "사기 의심"),
    ("중고 모니터를 사고 싶은데 화면 상태를 볼 수 있을까요?", "뚜렷한 위험 신호 없음"),
    ("채용 담당자가 면접 장소를 회사 홈페이지에 공지했대.", "뚜렷한 위험 신호 없음"),
    ("친구가 빌려준 돈을 돌려받았다고 메시지를 보냈어.", "뚜렷한 위험 신호 없음"),
    ("배송 조회는 내가 주문한 쇼핑몰의 주문 내역에서 확인했어.", "뚜렷한 위험 신호 없음"),
    ("우리 카카오톡에서 만날 장소만 정하고 결제는 물건을 본 뒤 거래 앱에서 하자.", "뚜렷한 위험 신호 없음"),
    ("증권사 안내문에 투자 원금이 보장되지 않는다고 쓰여 있었어.", "뚜렷한 위험 신호 없음"),
    ("판매자가 다른 연락 수단이 편하다고 하는데 결제 방법은 아직 말하지 않았어.", "판단 어려움"),
    ("문자로 법원 서류가 도착했다는 말만 들었어.", "판단 어려움"),
    ("그 사람이 급한 사정이 생겼다는데 무슨 일인지는 아직 몰라.", "판단 어려움"),
    ("계좌와 관련해 확인할 내용이 있다며 연락이 왔어.", "판단 어려움"),
    ("구매하고 싶은데 카톡으로 이동해서 결제할 수 있는지 물어봤어.", "판단 어려움"),
    ("취업 절차에 추가 단계가 있다고만 안내받았어.", "판단 어려움"),
]


def main():
    texts, labels = load_data()
    cases, expected = zip(*CASES)
    inputs = [format_input("", text) for text in cases]
    if set(texts) & set(inputs):
        raise ValueError("시험 문장이 학습 CSV에도 있습니다. 중복을 제거하세요.")
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(texts, labels)
    predicted = model.predict(inputs)
    order = ["사기 의심", "판단 어려움", "뚜렷한 위험 신호 없음"]
    print(f"학습 {len(texts)}개 / 별도 시험 {len(cases)}개 / 시험 판정별 {dict(Counter(expected))}")
    print(classification_report(expected, predicted, labels=order, zero_division=0))
    print("행=정답, 열=예측 / 순서:", order)
    print(confusion_matrix(expected, predicted, labels=order))
    print("\n오답:")
    errors = [(true, guess, text) for text, true, guess in zip(cases, expected, predicted) if true != guess]
    for true, guess, text in errors:
        print(f"- 정답={true} / 예측={guess} / {text}")
    if not errors:
        print("없음 (이 작은 시험만 통과한 것이며 실제 성능 검증은 아님)")


if __name__ == "__main__":
    main()
