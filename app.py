
from dotenv import load_dotenv
import os
import streamlit as st

# 環境変数の読み込み
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# 設定定数
MODEL_NAME = "openai:gpt-4o-mini"
TEMPERATURE = 0.5
EXPERTS = {
	"野球の専門家": "あなたは野球の専門家です。専門的で実践的な回答をしてください。必要に応じて用語の説明を加えてください。",
	"料理の専門家": "あなたは料理の専門家です。専門的で実践的な回答をしてください。必要に応じて用語の説明を加えてください。",
}

def query_expert(input_text: str, expert_choice: str) -> str:
	"""入力テキストと専門家選択を受け取り、LLM の回答を文字列で返す。

	Args:
		input_text: ユーザーが入力した質問テキスト
		expert_choice: 選択された専門家（"野球の専門家" または "料理の専門家"）

	Returns:
		LLM からの回答文字列

	Raises:
		RuntimeError: LangChain のインポートに失敗した場合
	"""
	# 動的に langchain の現行 API をインポート
	try:
		from langchain.chat_models import init_chat_model
	except Exception as e:
		raise RuntimeError(f"LangChain のインポートに失敗しました。`pip install langchain langchain-openai` を確認してください。詳細: {e}")

	# 選択された専門家に応じたシステムプロンプトを取得
	system_prompt = EXPERTS.get(expert_choice, EXPERTS["野球の専門家"])

	# LLM チャットモデルを初期化
	chat = init_chat_model(MODEL_NAME, temperature=TEMPERATURE)

	# プロンプトを送信し、回答を取得
	response = chat.invoke(f"{system_prompt}\n\n{input_text}")

	# 戻り値の型に応じて文字列に変換
	if isinstance(response, str):
		return response
	if hasattr(response, "content"):
		return response.content
	if hasattr(response, "generations"):
		try:
			return response.generations[0].message.content
		except (IndexError, AttributeError):
			pass
	return str(response)

st.set_page_config(page_title="２種の専門家に聞く", layout="centered")

st.title("２種の専門家に聞いてみよう")

st.write("専門家を選択して、質問を入力してください。LLM が専門家として回答します。")

# 専門家選択ラジオボタン
expert = st.radio(
	"選択する専門家を選んでください",
	list(EXPERTS.keys()),
)

with st.form("question_form"):
	question = st.text_area("質問を入力してください", height=100)
	submit = st.form_submit_button("質問する")

answer_slot = st.empty()

if submit:
 	answer_slot.header("回答")
 	answer_slot.write(f"**選択された専門家:** {expert}")
 	if not question or question.strip() == "":
 		answer_slot.warning("質問を入力してください。")
 	else:
			try:
				with st.spinner("LLM に問い合わせ中..."):
					resp = query_expert(question, expert)
					answer_slot.write(resp)
			except RuntimeError as e:
				answer_slot.error(str(e))
			except Exception as e:
				answer_slot.error(f"LLM への問い合わせ中にエラーが発生しました: {e}")
else:
 	answer_slot.header("回答")
 	answer_slot.write("質問して回答を表示します。")

st.markdown("---")

st.header("概要")
st.write(
	"このアプリは野球の専門家と料理の専門家の2種類の専門家を選択して、"
	"あなたの疑問や質問に専門的な回答をしてもらうことができます。"
)

st.header("操作方法")
st.write("""
1. 上のラジオボタンで「野球の専門家」または「料理の専門家」を選択します。
2. 下のテキストエリアに質問を入力します。
3. 「質問する」ボタンを押すと、選択された専門家としての回答が表示されます。
""")

st.header("注意事項")
st.write(
	"生成AIの回答は必ずしも正確ではないため、参考情報としてご利用ください。"
	"重要な判断は実際の専門家にご相談ください。"
)