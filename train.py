"""낌새 판정 모델: CSV의 모든 문장으로 학습하고 새 문장을 분류한다.

사용법: python train.py
       python train.py "카드가 배송됩니다" --situation "카드를 신청한 적이 없습니다"

이 파일은 시험 점수를 출력하지 않는다. 새 데이터로 성능을 평가해야 한다.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


DATA_PATH = Path(__file__).with_name("낌새_AI분석_상황포함_530개_학습후보.csv")
LABELS = {"사기 의심", "판단 어려움", "뚜렷한 위험 신호 없음"}
# 사기 판정 3종 밖의 명시적인 신체 위협은 별도로 표시한다.
DIRECT_KIDNAP_THREAT = re.compile(
    r"(?:납치|유괴)\s*(?:해도|할|했다|했어|하겠|하러|하려|한다|한다면|당할|당했다|당했어)"
)
CHILD_TAKE_THREAT = re.compile(r"(?:아이|애|자녀|아들|딸).{0,16}데려가(?:도|겠|려고|고\s*싶)")
KIDNAPPING_CONTEXT = re.compile(r"납치|유괴|감금|아이를\s*데려갔|자녀를\s*데려갔")
MONEY_DEMAND = re.compile(r"입금|송금|돈을\s*보내|현금|몸값")
CHILD_MONEY_THREAT = re.compile(
    r"(?:아이|자녀|아들|딸).{0,50}(?:해치|죽이|어떻게\s*되는지|무사하지\s*못)"
    r"|(?:입금하지\s*않으면|돈을\s*보내지\s*않으면|몸값).{0,60}(?:아이|자녀|아들|딸).{0,40}(?:어떻게|해치|죽이)"
)

# 판정 기준: 발화의 요청과 사용자가 설명한 정황을 함께 본다.
# 이 규칙들은 사기를 확정하지 않는다. 명백한 위험 신호가 있거나
# 확인되지 않은 금전 요청이 있을 때 '위험 신호 없음'을 방지한다.
PAYMENT_REQUEST = re.compile(
    r"(?:돈|비용|항공료|병원비|투자금|보증금|수수료|세금|대출금|위약금|원|만원).{0,25}"
    r"(?:줘|주세요|보내|입금|송금|이체|내야|납부|결제|빌려|필요|요구)"
    r"|(?:입금|송금|이체|결제|빌려|돈을\s*줘|돈을\s*요구).{0,25}"
    r"(?:돈|비용|원|만원|계좌|나에게|제게|줘|주세요|해|하|해요|해야|부탁)"
)
PAYMENT_CONTEXT = re.compile(r"(?:돈|비용|항공료|병원비|투자금|보증금|수수료|세금).{0,25}(?:요구|보내|달라|필요|입금|송금|빌려)")
CREDENTIAL_REQUEST = re.compile(
    r"(?:비밀번호|인증번호|보안카드|주민등록번호|신분증\s*사진|카드번호).{0,30}"
    r"(?:입력|알려|보내|전송|제출|찍어|등록)"
    r"|(?:입력|알려|보내|전송|제출).{0,30}(?:비밀번호|인증번호|보안카드|주민등록번호|신분증\s*사진|카드번호)"
)
OUTSIDE_CHANNEL = re.compile(r"라인|텔레그램|오픈채팅|외부\s*(?:주소|링크|사이트)|문자\s*(?:링크|주소)|낯선\s*링크")
IMPOSSIBLE_REWARD = re.compile(r"(?:한\s*건|리뷰\s*(?:한\s*건)?|댓글\s*(?:한\s*건)?|클릭\s*(?:한\s*번)?).{0,25}(?:100\s*만\s*원|수십\s*만\s*원|고수익|큰돈|고액)")
JOB_PITCH = re.compile(r"(?:고수익|고액|쉽게\s*돈|확정\s*수익|수익\s*보장).{0,35}(?:알바|부업|리뷰|투자|돈|수익)|(?:알바|부업|리뷰).{0,35}(?:고수익|고액|100\s*만\s*원|수익\s*보장)")
COERCION = re.compile(r"(?:입금|송금|납부|결제).{0,35}(?:안\s*하면|않으면|하지\s*않으면)|(?:안\s*하면|않으면|하지\s*않으면).{0,35}(?:입금|송금|납부|결제)")
VERIFIED = re.compile(r"공식\s*(?:앱|사이트|홈페이지).{0,55}(?:직접|확인|내역|일치)|(?:직접|내역).{0,40}공식\s*(?:앱|사이트|홈페이지)")
SUSPICIOUS_PAYMENT = re.compile(r"(?:개인\s*계좌|처음\s*보는\s*계좌|상대를\s*사칭|연인을\s*사칭|가족을\s*사칭).{0,70}(?:송금|입금|돈|비용|이체)|(?:송금|입금|이체).{0,70}(?:개인\s*계좌|처음\s*보는\s*계좌|사칭)")
NO_PAYMENT_REQUEST = re.compile(r"(?:돈|비용|투자|송금|입금).{0,15}(?:요구|요청|부탁)(?:는|가)?\s*(?:없|하지\s*않)")
# 확실히 일상적인 초대 발화에만 적용한다. '위험 단어가 없다'는
# 사실만으로 모든 입력을 안전하다고 판정하지 않도록 범위를 좁혔다.
SOCIAL_INVITATION = re.compile(
    r"^(?:[가-힣]{1,8}[아야],?\s*)?(?:오늘|내일|주말에\s*)?"
    r"(?:술\s*마실래|밥\s*먹을래|커피\s*마실래|영화\s*볼래|산책\s*할래)\s*\?$"
)
FRAUD_CONTEXT = re.compile(
    r"송금|입금|이체|결제|대출|투자|수익|알바|부업|계좌|인증|비밀번호|"
    r"링크|앱\s*설치|카드번호|신분증|사칭|협박|납치|유괴|몸값|"
    r"(?:돈|비용|\d+\s*(?:원|만원)).{0,25}(?:요구|보내|달라|필요|빌려)"
)


def immediate_threat(situation, text):
    # 단순히 '납치 관련 뉴스를 봤다'처럼 범죄를 언급한 문장은 여기서 제외한다.
    if DIRECT_KIDNAP_THREAT.search(text):
        return True
    if CHILD_TAKE_THREAT.search(text) and re.search(r"낯선|모르는|처음\s*보는", situation):
        return True
    if KIDNAPPING_CONTEXT.search(situation) and MONEY_DEMAND.search(text):
        return True
    return bool(CHILD_MONEY_THREAT.search(text))


def format_input(situation, text):
    return f"상황: {situation.strip() or '추가 상황 설명 없음'}\n받은 말: {text.strip()}"


def read_rows():
    # Excel에서 저장한 한국어 CSV는 CP949일 수 있다.
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with DATA_PATH.open(encoding=encoding, newline="") as file:
                rows = list(csv.DictReader(file))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장하세요.")

    if not rows or any(
        not row["text"].strip() or not row["situation"].strip() or row["judgment"] not in LABELS
        for row in rows
    ):
        raise ValueError("CSV의 text, situation 또는 judgment 열을 확인하세요.")
    return rows


def load_data():
    rows = read_rows()
    return [format_input(row["situation"], row["text"]) for row in rows], [row["judgment"] for row in rows]


def load_pairs():
    rows = read_rows()
    return [[row["situation"], row["text"]] for row in rows], [row["judgment"] for row in rows]


def build_model():
    # 상황과 발화를 따로 벡터화하여 같은 말의 다른 상황을 비교할 수 있게 한다.
    features = ColumnTransformer([
        ("situation", TfidfVectorizer(analyzer="char", ngram_range=(2, 4)), 0),
        ("text", TfidfVectorizer(analyzer="char", ngram_range=(2, 4)), 1),
    ])
    return make_pipeline(features, LogisticRegression(max_iter=1000, class_weight="balanced"))


def build_text_only_model():
    # 상황 정보가 없을 때는 합쳐 쓰던 기존 문장 분류 방식을 사용한다.
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def assess_signals(situation, text):
    """Return a conservative label and reason, or None to defer to the model."""
    context = f"{situation} {text}"
    if immediate_threat(situation, text):
        return "긴급 위협 의심", "신체적 위협 표현이 감지됨"
    if CREDENTIAL_REQUEST.search(text) and (
        OUTSIDE_CHANNEL.search(context) or re.search(r"링크|문자|낯선|모르는|처음", context)
    ):
        return "사기 의심", "외부 경로에서 민감정보를 요구함"
    if IMPOSSIBLE_REWARD.search(text) or (JOB_PITCH.search(text) and OUTSIDE_CHANNEL.search(context)):
        return "사기 의심", "비현실적인 수익 또는 외부 메신저의 고수익 제안"
    if COERCION.search(text) and PAYMENT_REQUEST.search(context):
        return "사기 의심", "금전 요구와 압박이 함께 나타남"
    if SUSPICIOUS_PAYMENT.search(context):
        return "사기 의심", "신원 사칭 또는 개인 계좌를 통한 금전 요구"
    if (PAYMENT_REQUEST.search(text) or (PAYMENT_CONTEXT.search(situation) and not NO_PAYMENT_REQUEST.search(situation))) and not VERIFIED.search(situation):
        return "판단 어려움", "금전 요구가 있지만 상대와 사유를 독립적으로 확인하지 못함"
    return None


def predict_case(model, situation, text):
    """Apply the stated decision order, then use the trained model if no rule applies."""
    signal = assess_signals(situation, text)
    if signal is not None and signal[0] in {"긴급 위협 의심", "사기 의심"}:
        return signal
    if SOCIAL_INVITATION.fullmatch(text.strip()) and not FRAUD_CONTEXT.search(f"{situation} {text}"):
        return "뚜렷한 위험 신호 없음", "제공된 내용은 일상적인 초대이며 사기 관련 요구가 보이지 않음"
    if situation.strip():
        guess = str(model.predict([[situation, text]])[0])
    else:
        guess = str(model.predict([format_input('', text)])[0])
    # 정보가 부족한 금전 요청만 정상으로 빠지지 않게 한다. 모델이 이미
    # 사기를 의심하면 그 결과를 약한 '판단 어려움' 규칙으로 덮지 않는다.
    if signal is not None and guess == "뚜렷한 위험 신호 없음":
        return signal
    return guess, "학습 모델의 예측; 강한 위험 신호 규칙에는 해당하지 않음"


def main():
    parser = argparse.ArgumentParser(description="낌새 문장 판정 실험")
    parser.add_argument("text", nargs="?", help="분석할 새 문장 (생략하면 터미널에서 입력)")
    parser.add_argument("--situation", default="", help="사용자가 설명한 상황")
    args = parser.parse_args()

    sentence = args.text if args.text is not None else input("분석할 문장 (엔터만 누르면 종료): ")
    if sentence.strip():
        context = args.situation or (input("상황 설명 (없으면 엔터): ") if args.text is None else "")
        if context.strip():
            pairs, judgments = load_pairs()
            model = build_model().fit(pairs, judgments)
        else:
            texts, judgments = load_data()
            model = build_text_only_model().fit(texts, judgments)
        print(f"학습 완료: {len(judgments)}개 / 판정별: {dict(Counter(judgments))}")
        result, reason = predict_case(model, context, sentence)
        print(f"예측: {result}")
        print(f"근거: {reason}")
        if result == "긴급 위협 의심":
            print("실제 상황이라면 아이와 안전한 곳으로 이동하고 즉시 112에 신고하세요.")


if __name__ == "__main__":
    main()
